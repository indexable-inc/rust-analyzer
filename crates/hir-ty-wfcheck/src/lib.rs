//! Item well-formedness: the part of rustc's `wfcheck` (`rustc_hir_analysis::check::wfcheck`)
//! about what outlives what, which reports E0309 and E0310 outside function bodies.
//!
//! An ADT's fields must be well-formed given its where clauses and what rustc infers from
//! the fields themselves (RFC 2093), which is every `T: 'a` and never `T: 'static`. A trait
//! impl's associated types must be well-formed given the impl's where clauses and what its
//! header's types imply, and the impl must satisfy the outlives bounds of the trait, as
//! `trait Foo: 'static`. Each item is a query of its own, as the borrow checker's bodies are;
//! `docs/borrowck.md` in the repository describes what is checked.

extern crate ra_ap_rustc_type_ir as rustc_type_ir;

mod outlives;

use hir_def::{
    AdtId, AssocItemId, GenericDefId, ImplId, LocalFieldId, TypeAliasId, VariantId,
    hir::generics::GenericParams,
};
use rustc_type_ir::{RegionKind, TypeVisitableExt as _, inherent::IntoKind};
use salsa::SalsaValue;

use hir_ty::{
    GenericPredicates,
    db::HirDatabase,
    next_solver::{Clause, DbInterner, GenericArgKind, Ty, TyKind},
};

use outlives::{Inferred, Known, Needs};

/// Where an error is: rustc's span of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, SalsaValue)]
pub enum Site {
    /// The type of a field.
    Field(VariantId, LocalFieldId),
    /// The type an impl gives an associated type.
    AssocType(TypeAliasId),
    /// The trait an impl implements.
    ImplTrait(ImplId),
}

#[derive(Debug, Clone, PartialEq, Eq, SalsaValue)]
pub struct WfDiagnostic {
    /// rustc's error code, as `E0310`.
    pub code: &'static str,
    pub message: String,
    pub site: Site,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, SalsaValue)]
pub struct WfResult {
    pub diagnostics: Vec<WfDiagnostic>,
}

/// The well-formedness errors of the item `def`: an ADT or an impl.
#[salsa::tracked(returns(ref))]
pub fn wfcheck(db: &dyn HirDatabase, def: GenericDefId) -> WfResult {
    let mut result = WfResult::default();
    if matches!(std::env::var("RA_JIT_BORROWCK").as_deref(), Ok("off" | "0")) {
        return result;
    }
    let mut check = Check { db, inferred: Inferred::default(), found: &mut result.diagnostics };
    match def {
        GenericDefId::AdtId(adt) => check.adt(adt),
        GenericDefId::ImplId(impl_) => check.impl_(impl_),
        _ => (),
    }
    result
}

struct Check<'a, 'db> {
    db: &'db dyn HirDatabase,
    inferred: Inferred<'db>,
    found: &'a mut Vec<WfDiagnostic>,
}

impl<'db> Check<'_, 'db> {
    fn interner(&self) -> DbInterner<'db> {
        DbInterner::new_no_crate(self.db)
    }

    fn clauses(&self, def: GenericDefId) -> Vec<Clause<'db>> {
        self.db.trait_environment(def).clauses().iter().collect()
    }

    /// Every field must be well-formed. What it needs of a lifetime parameter rustc infers,
    /// so only what it needs of `'static` can fail.
    fn adt(&mut self, adt: AdtId) {
        let known = Known::new(&self.clauses(adt.into()));
        let mut needs = Vec::new();
        for variant in outlives::variants(self.db, adt) {
            for (field, ty) in self.db.field_types(variant).iter() {
                let ty = ty.ty().instantiate_identity().skip_norm_wip();
                if ty.references_error() {
                    continue;
                }
                let mut found = Vec::new();
                outlives::wf(self.db, &mut self.inferred, ty, &mut found);
                needs.extend(
                    found
                        .into_iter()
                        .filter(|it| match *it {
                            Needs::Type(_, region) | Needs::Region(_, region) => {
                                matches!(region.kind(), RegionKind::ReStatic)
                            }
                        })
                        .map(|it| (it, Site::Field(variant, field))),
                );
            }
        }
        self.resolve(&known, needs);
    }

    fn impl_(&mut self, impl_: ImplId) {
        let db = self.db;
        let Some(trait_ref) = db.impl_trait(impl_) else { return };
        let trait_ref = trait_ref.instantiate_identity().skip_norm_wip();
        let self_ty = db.impl_self_ty(impl_).instantiate_identity().skip_norm_wip();
        let header: Vec<Ty<'db>> = trait_ref
            .args
            .iter()
            .filter_map(|arg| match arg.kind() {
                GenericArgKind::Type(ty) => Some(ty),
                _ => None,
            })
            .chain([self_ty])
            .collect();
        // What normalizing the header would imply, or an error in it, is not known here.
        if header.iter().any(|ty| ty.references_error() || ty.has_aliases()) {
            return;
        }
        let mut known = Known::new(&self.clauses(impl_.into()));
        let mut implied = Vec::new();
        for &ty in &header {
            outlives::wf(db, &mut self.inferred, ty, &mut implied);
        }
        known.add(implied);

        // The trait's outlives bounds hold of the impl's arguments.
        let trait_clauses: Vec<Clause<'db>> =
            GenericPredicates::query_all(db, trait_ref.def_id.0.into())
                .iter_instantiated(self.interner(), trait_ref.args.as_slice())
                .map(|it| it.skip_norm_wip())
                .collect();
        let mut needs = Vec::new();
        outlives::clause_needs(&trait_clauses, &mut needs);
        needs.retain(|it| match *it {
            Needs::Type(ty, region) => {
                !ty.has_escaping_bound_vars() && !region.has_escaping_bound_vars()
            }
            Needs::Region(longer, shorter) => {
                !longer.has_escaping_bound_vars() && !shorter.has_escaping_bound_vars()
            }
        });
        self.resolve(&known, needs.into_iter().map(|it| (it, Site::ImplTrait(impl_))).collect());

        // Each associated type is well-formed.
        for &(_, item) in impl_.impl_items(db).items.iter() {
            let AssocItemId::TypeAliasId(alias) = item else { continue };
            // A generic associated type assumes what the trait's where clauses on it say,
            // which is not reasoned about here.
            if !GenericParams::of(db, alias.into()).is_empty() {
                continue;
            }
            let ty = db.ty(alias.into()).instantiate_identity().skip_norm_wip();
            if ty.references_error() {
                continue;
            }
            let mut needs = Vec::new();
            outlives::wf(db, &mut self.inferred, ty, &mut needs);
            self.resolve(
                &known,
                needs.into_iter().map(|it| (it, Site::AssocType(alias))).collect(),
            );
        }
    }

    /// E0309 or E0310 for each type parameter not known to outlive a region it must, unless a
    /// region must outlive one it is not known to: rustc's `process_errors` keeps only that
    /// error of one check, as the others follow from it.
    fn resolve(&mut self, known: &Known<'db>, needs: Vec<(Needs<'db>, Site)>) {
        let region_fails = needs.iter().any(|&(need, _)| match need {
            Needs::Region(longer, shorter) => !known.region_outlives(longer, shorter),
            Needs::Type(ty, region) => outlives::components(ty).into_iter().any(|it| {
                matches!(it, outlives::Component::Region(longer)
                    if !known.region_outlives(longer, region))
            }),
        });
        if region_fails {
            return;
        }
        for (need, site) in needs {
            let Needs::Type(ty, region) = need else { continue };
            for param in known.failing_params(ty, region) {
                let TyKind::Param(param) = param.kind() else { continue };
                let code = match region.kind() {
                    RegionKind::ReStatic => "E0310",
                    _ => "E0309",
                };
                let params = GenericParams::of(self.db, param.id.parent());
                let name = params[param.id.local_id()].name().map_or("_", |it| it.as_str());
                let message = format!("the parameter type `{name}` may not live long enough");
                let diagnostic = WfDiagnostic { code, message, site };
                if !self.found.contains(&diagnostic) {
                    self.found.push(diagnostic);
                }
            }
        }
    }
}
