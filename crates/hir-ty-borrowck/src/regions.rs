//! Regions, as rustc's `renumber.rs`, `universal_regions.rs`, `type_check` and
//! `region_infer` in `rustc_borrowck`.
//!
//! Every region in the types of a body becomes a variable. The regions of the signature are
//! the universal ones: they hold on all of the body and past its end. Relating the types each
//! statement puts together says which regions outlive which; liveness says at which points a
//! region must hold; propagating both gives each region the points it holds at, which is
//! where a loan of that region is in scope. A universal region made to outlive another the
//! signature does not say it outlives is rustc's "lifetime may not live long enough".

use hir_def::{
    AdtId, CallableDefId, DefWithBodyId, GenericDefId, ItemContainerId, Lookup as _,
    hir::{Pat, Statement},
    signatures::FunctionSignature,
};
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{
    ClauseKind, OutlivesPredicate, RegionKind, Variance,
    inherent::{GenericArgs as _, IntoKind, ParamEnv as _, SliceLike as _, Ty as _},
};

use hir_ty::{
    GenericPredicates, InferBodyId,
    mir::{
        AggregateKind, BorrowKind, CastKind, LocalId, MirBody, MirSpan, Operand, OperandKind,
        PlaceElem, PlaceTy, ProjectionElem, Rvalue, StatementKind, TerminatorKind,
    },
    next_solver::{GenericArgKind, GenericArgs, Region, Ty, TyKind},
};

use super::{
    Ctx,
    bitset::BitSet,
    cfg::{Cfg, Location},
};

/// A region variable.
pub(super) type Rv = usize;

/// Why one region outlives another, as rustc's `ConstraintCategory`; the most interesting
/// first, which is the order errors pick the constraint to point at by.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub(super) enum Category {
    Return,
    TypeAnnotation,
    Cast,
    CallArgument,
    Assignment,
    Usage,
    Predicate,
    Boring,
    Internal,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Outlives {
    pub(super) sup: Rv,
    pub(super) sub: Rv,
    pub(super) location: Option<Location>,
    pub(super) category: Category,
    pub(super) span: MirSpan,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum UniversalKind {
    Static,
    /// The body itself: every universal region outlives it.
    FnBody,
    /// A lifetime parameter.
    Named,
    /// An elided lifetime of a parameter.
    AnonParam,
    /// Any other region of the signature, as an elided lifetime of the return type that
    /// elision gives no lifetime to.
    Anon,
}

/// That the type parameter `ty` must outlive `lower`, as rustc's `TypeTest`.
pub(super) struct TypeTest<'db> {
    pub(super) ty: Ty<'db>,
    pub(super) lower: Rv,
    pub(super) span: MirSpan,
}

pub(super) struct Regions<'db> {
    pub(super) universals: Vec<(Rv, UniversalKind)>,
    universal_index: FxHashMap<Rv, usize>,
    pub(super) static_rv: Rv,
    /// The universal region of each early-bound lifetime parameter, by its index.
    pub(super) early: FxHashMap<u32, Rv>,
    /// Per universal region, the universal regions it is known to outlive.
    known: Vec<BitSet>,
    /// The type parameters known to outlive a region.
    type_bounds: Vec<(Ty<'db>, Rv)>,
    /// The type of each local, its regions variables.
    pub(super) local_tys: Vec<Ty<'db>>,
    /// The region of the loan each borrow makes.
    pub(super) loan_regions: FxHashMap<Location, Rv>,
    pub(super) outlives: Vec<Outlives>,
    pub(super) type_tests: Vec<TypeTest<'db>>,
    /// Whether universal regions are checked against each other: not in a closure, whose
    /// regions rustc relates to its parent's, which this does not do yet.
    pub(super) check_universals: bool,
    pub(super) is_impl_item: bool,
}

impl<'db> Regions<'db> {
    pub(super) fn universal(&self, rv: Rv) -> Option<usize> {
        self.universal_index.get(&rv).copied()
    }

    pub(super) fn knows(&self, longer: usize, shorter: usize) -> bool {
        self.known[longer].contains(shorter)
    }

    /// The regions the where clauses say the type parameter `ty` outlives.
    pub(super) fn type_bounds_of(&self, ty: Ty<'db>) -> Vec<Rv> {
        self.type_bounds
            .iter()
            .filter(|&&(bounded, _)| bounded == ty)
            .map(|&(_, region)| region)
            .collect()
    }

    /// Whether the type parameter `ty` is known to outlive the universal region `universal`.
    pub(super) fn type_outlives_known(&self, ty: Ty<'_>, universal: usize) -> bool {
        self.type_bounds.iter().any(|&(bounded, region)| {
            bounded == ty && self.universal(region).is_some_and(|it| self.knows(it, universal))
        })
    }
}

pub(super) fn rv_of(region: Region<'_>) -> Option<Rv> {
    match region.kind() {
        RegionKind::ReVar(vid) => Some(vid.as_usize()),
        _ => None,
    }
}

/// Each region in `ty`, in order, those under a binder aside.
/// The associated types `ty` names, with their arguments.
fn projections<'db>(ty: Ty<'db>) -> Vec<(hir_ty::next_solver::SolverDefId<'db>, GenericArgs<'db>)> {
    struct Find<'db>(Vec<(hir_ty::next_solver::SolverDefId<'db>, GenericArgs<'db>)>);
    impl<'db> rustc_type_ir::TypeVisitor<hir_ty::next_solver::DbInterner<'db>> for Find<'db> {
        type Result = ();
        fn visit_ty(&mut self, ty: Ty<'db>) {
            if let TyKind::Alias(alias) = ty.kind()
                && let rustc_type_ir::AliasTyKind::Projection { .. } = alias.kind
            {
                self.0.push((alias.kind.def_id(), alias.args));
            }
            rustc_type_ir::TypeSuperVisitable::super_visit_with(&ty, self)
        }
    }
    let mut find = Find(Vec::new());
    rustc_type_ir::TypeVisitable::visit_with(&ty, &mut find);
    find.0
}

/// Whether `ty` has an `impl Trait` in it.
fn has_opaque(ty: Ty<'_>) -> bool {
    struct Find;
    impl<'db> rustc_type_ir::TypeVisitor<hir_ty::next_solver::DbInterner<'db>> for Find {
        type Result = std::ops::ControlFlow<()>;
        fn visit_ty(&mut self, ty: Ty<'db>) -> Self::Result {
            if let TyKind::Alias(alias) = ty.kind()
                && let rustc_type_ir::AliasTyKind::Opaque { .. } = alias.kind
            {
                return std::ops::ControlFlow::Break(());
            }
            rustc_type_ir::TypeSuperVisitable::super_visit_with(&ty, self)
        }
    }
    rustc_type_ir::TypeVisitable::visit_with(&ty, &mut Find).is_break()
}

pub(super) fn regions_of(ty: Ty<'_>, out: &mut Vec<Rv>) {
    struct Collect<'o>(&'o mut Vec<Rv>);
    impl<'db> rustc_type_ir::TypeVisitor<hir_ty::next_solver::DbInterner<'db>> for Collect<'_> {
        type Result = ();
        fn visit_region(&mut self, region: Region<'db>) {
            if let Some(rv) = rv_of(region) {
                self.0.push(rv);
            }
        }
    }
    rustc_type_ir::TypeVisitable::visit_with(&ty, &mut Collect(out));
}

pub(super) fn build<'db>(ctx: &Ctx<'_, 'db>, body: &MirBody<'db>, cfg: &Cfg) -> Regions<'db> {
    let mut builder = Builder {
        ctx,
        body,
        regions: Regions {
            universals: Vec::new(),
            universal_index: FxHashMap::default(),
            static_rv: 0,
            known: Vec::new(),
            type_bounds: Vec::new(),
            early: FxHashMap::default(),
            local_tys: Vec::new(),
            loan_regions: FxHashMap::default(),
            outlives: Vec::new(),
            type_tests: Vec::new(),
            check_universals: ctx.closure_local.is_none(),
            is_impl_item: false,
        },
        early: FxHashMap::default(),
        late: FxHashMap::default(),
        impl_self: None,
        returns_opaque: false,
        location: None,
    };
    builder.universal_regions();
    builder.location = None;
    builder.annotations();
    for &block in &cfg.order {
        let data = &body.basic_blocks[block];
        for (statement, it) in data.statements.iter().enumerate() {
            builder.location = Some(Location { block, statement });
            builder.statement(&it.kind, it.span);
        }
        if let Some(terminator) = &data.terminator {
            builder.location = Some(Location { block, statement: data.statements.len() });
            builder.terminator(&terminator.kind, terminator.span);
        }
    }
    // Last, as annotations and turbofishes can name lifetimes the signature does not.
    builder.known_relations();
    builder.regions
}

struct Builder<'c, 'a, 'db> {
    ctx: &'c Ctx<'a, 'db>,
    body: &'c MirBody<'db>,
    regions: Regions<'db>,
    /// The universal region of each lifetime parameter, by its index.
    early: FxHashMap<u32, Region<'db>>,
    /// The universal region of each of the signature's late-bound lifetimes.
    late: FxHashMap<u32, Region<'db>>,
    /// The self type of the impl the body is in, as rust-analyzer gives it with `'_` an error,
    /// and with universal regions there: rustc's anonymous lifetime parameters of the impl.
    impl_self: Option<(Ty<'db>, Ty<'db>)>,
    /// The return type is an `impl Trait`, so the return slot has the type it hides, which
    /// implies nothing to the body.
    returns_opaque: bool,
    location: Option<Location>,
}

/// Where a constraint comes from.
#[derive(Clone, Copy)]
struct At {
    category: Category,
    span: MirSpan,
}

impl<'db> Builder<'_, '_, 'db> {
    fn interner(&self) -> hir_ty::next_solver::DbInterner<'db> {
        self.ctx.infcx.interner
    }

    fn fresh(&self) -> Region<'db> {
        self.ctx.infcx.next_region_var(hir_ty::Span::Dummy)
    }

    fn renumber<T: rustc_type_ir::TypeFoldable<hir_ty::next_solver::DbInterner<'db>>>(
        &self,
        value: T,
    ) -> T {
        let infcx = &self.ctx.infcx;
        rustc_type_ir::fold_regions(self.interner(), value, |region, _| match region.kind() {
            RegionKind::ReBound(..) => region,
            _ => infcx.next_region_var(hir_ty::Span::Dummy),
        })
    }

    fn new_universal(&mut self, kind: UniversalKind) -> Region<'db> {
        let region = self.fresh();
        let rv = rv_of(region).expect("a fresh region is a variable");
        self.regions.universal_index.insert(rv, self.regions.universals.len());
        self.regions.universals.push((rv, kind));
        region
    }

    fn static_region(&self) -> Region<'db> {
        Region::new_var(
            self.interner(),
            rustc_type_ir::RegionVid::from_usize(self.regions.static_rv),
        )
    }

    fn early_region(&mut self, index: u32) -> Region<'db> {
        if let Some(&it) = self.early.get(&index) {
            return it;
        }
        let region = self.new_universal(UniversalKind::Named);
        self.early.insert(index, region);
        self.regions.early.insert(index, rv_of(region).unwrap());
        region
    }

    /// `region` of the signature as a universal region; `None` for an elided one. `depth` is
    /// how many binders in the region is, those of the signature's own late-bound lifetimes
    /// being outside it.
    fn signature_region(
        &mut self,
        region: Region<'db>,
        depth: rustc_type_ir::DebruijnIndex,
    ) -> Option<Region<'db>> {
        match region.kind() {
            RegionKind::ReBound(rustc_type_ir::BoundVarIndexKind::Bound(debruijn), bound)
                if debruijn == depth =>
            {
                let var = bound.var.as_u32();
                if let Some(&it) = self.late.get(&var) {
                    return Some(it);
                }
                let kind = match bound.kind {
                    rustc_type_ir::BoundRegionKind::Named(_) => UniversalKind::Named,
                    _ => UniversalKind::AnonParam,
                };
                let region = self.new_universal(kind);
                self.late.insert(var, region);
                Some(region)
            }
            RegionKind::ReBound(..) => Some(region),
            // A late-bound lifetime the body names, the same as in the signature.
            RegionKind::ReLateParam(late) if self.ctx.closure_local.is_none() => {
                let var = late.bound_region.var.as_u32();
                if let Some(&it) = self.late.get(&var) {
                    return Some(it);
                }
                let region = self.new_universal(UniversalKind::Named);
                self.late.insert(var, region);
                Some(region)
            }
            RegionKind::ReStatic => Some(self.static_region()),
            RegionKind::ReEarlyParam(early) => Some(self.early_region(early.index)),
            RegionKind::ReVar(_) => Some(region),
            _ => None,
        }
    }

    fn universal_regions(&mut self) {
        let static_region = self.new_universal(UniversalKind::Static);
        self.regions.static_rv = rv_of(static_region).unwrap();
        self.new_universal(UniversalKind::FnBody);
        let body = self.body;
        let mut local_tys: Vec<Option<Ty<'db>>> = vec![None; body.locals.len()];
        let return_local = super::return_slot();
        let owner = body.owner;
        if self.ctx.closure_local.is_some() {
            for &local in body.param_locals.iter().chain([&return_local]) {
                let ty = body.locals[local].ty.as_ref();
                let ty = rustc_type_ir::fold_regions(self.interner(), ty, |region, _| match region
                    .kind()
                {
                    RegionKind::ReBound(..) => region,
                    _ => self.new_universal(UniversalKind::AnonParam),
                });
                local_tys[local.into_raw().into_u32() as usize] = Some(ty);
            }
        } else {
            // The declared signature, as the parameters' local types have their lifetimes
            // erased.
            let signature = match owner {
                InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(function)) => {
                    let sig = self
                        .ctx
                        .db
                        .callable_item_signature(CallableDefId::FunctionId(function))
                        .instantiate_identity()
                        .skip_norm_wip()
                        .skip_binder();
                    let mut inputs: Vec<Ty<'db>> = sig.inputs().iter().copied().collect();
                    let mut output = sig.output();
                    let written = hir_def::signatures::FunctionSignature::of(self.ctx.db, function);
                    if written.params.len() == inputs.len()
                        && let Some(impl_self) = self.impl_self_of(function)
                    {
                        // Only where `Self` is written: an elided `&str` in `impl Trait for &str`
                        // is the same type to rust-analyzer, but a lifetime of its own.
                        let store = &written.store;
                        let mut folder = ImplSelf { interner: self.interner(), impl_self };
                        for (input, &param) in inputs.iter_mut().zip(&written.params) {
                            if writes_self(store, param) {
                                *input =
                                    rustc_type_ir::TypeFoldable::fold_with(*input, &mut folder);
                            }
                        }
                        if written.ret_type.is_some_and(|it| writes_self(store, it)) {
                            output = rustc_type_ir::TypeFoldable::fold_with(output, &mut folder);
                        }
                    }
                    // Each elided input lifetime is its own region before `&dyn Trait` copies
                    // it into the object's default, so that the two are the same one.
                    for input in &mut inputs {
                        *input =
                            rustc_type_ir::fold_regions(self.interner(), *input, |region, _| {
                                match region.kind() {
                                    RegionKind::ReError(_) => {
                                        self.new_universal(UniversalKind::AnonParam)
                                    }
                                    _ => region,
                                }
                            });
                    }
                    let mut defaults = ObjectDefaults {
                        db: self.ctx.db,
                        interner: self.interner(),
                        keep_static: writes_static_object(&written.store),
                        erase: false,
                    };
                    for input in &mut inputs {
                        *input = rustc_type_ir::TypeFoldable::fold_with(*input, &mut defaults);
                    }
                    output = rustc_type_ir::TypeFoldable::fold_with(output, &mut defaults);
                    (inputs.len() == body.param_locals.len()).then_some((inputs, output))
                }
                _ => None,
            };
            let mut inputs = Vec::new();
            for (index, &local) in body.param_locals.iter().enumerate() {
                let ty = match &signature {
                    Some((tys, _)) => tys[index],
                    None => body.locals[local].ty.as_ref(),
                };
                let ty = rustc_type_ir::fold_regions(self.interner(), ty, |region, depth| {
                    // Inference variables are ours only in a signature, from its impl's self type.
                    let region = match region.kind() {
                        RegionKind::ReVar(_) if signature.is_none() => None,
                        _ => self.signature_region(region, depth),
                    }
                    .unwrap_or_else(|| self.new_universal(UniversalKind::AnonParam));
                    if !matches!(region.kind(), RegionKind::ReBound(..)) {
                        inputs.push(region);
                    }
                    region
                });
                local_tys[local.into_raw().into_u32() as usize] = Some(ty);
            }
            // The lifetime elision gives the elided lifetimes of the return type.
            let self_region = match owner {
                InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(function))
                    if FunctionSignature::of(self.ctx.db, function).has_self_param() =>
                {
                    body.param_locals.first().and_then(|&it| {
                        match local_tys[it.into_raw().into_u32() as usize]?.kind() {
                            TyKind::Ref(region, ..) => Some(region),
                            _ => None,
                        }
                    })
                }
                _ => None,
            };
            inputs.dedup();
            let elided = match (inputs.as_slice(), self_region) {
                (_, Some(region)) => Some(region),
                ([only], None) => Some(*only),
                _ => None,
            };
            let is_fn = matches!(owner, InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(_)));
            // A returned `impl Trait` is checked as the type it hides; its lifetimes are
            // `'static` when nothing it could capture is, and left free otherwise.
            let opaque = signature.as_ref().is_some_and(|(_, output)| has_opaque(*output));
            self.returns_opaque = opaque;
            let ty = match &signature {
                Some((_, output)) if !opaque => *output,
                _ => body.locals[return_local].ty.as_ref(),
            };
            let mut names_parameter = false;
            rustc_type_ir::fold_regions(self.interner(), ty, |region, _| {
                names_parameter |= matches!(
                    region.kind(),
                    RegionKind::ReEarlyParam(_) | RegionKind::ReLateParam(_)
                );
                region
            });
            let hidden_static = opaque && self.regions.universals.len() == 2 && !names_parameter;
            let ty = rustc_type_ir::fold_regions(self.interner(), ty, |region, depth| {
                if (opaque || signature.is_none()) && matches!(region.kind(), RegionKind::ReVar(_))
                {
                    return self.fresh();
                }
                self.signature_region(region, depth).unwrap_or_else(|| match (elided, is_fn) {
                    _ if hidden_static => self.static_region(),
                    _ if opaque => self.fresh(),
                    (Some(it), _) => it,
                    (None, true) => self.new_universal(UniversalKind::Anon),
                    // The elided lifetimes of the type of a constant or a static are `'static`.
                    (None, false) => self.static_region(),
                })
            });
            local_tys[return_local.into_raw().into_u32() as usize] = Some(ty);
            if let InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(function)) = owner {
                self.regions.is_impl_item =
                    matches!(function.lookup(self.ctx.db).container, ItemContainerId::ImplId(_));
            }
        }
        // A variable with a type written out keeps the lifetimes the type names, as rustc
        // relates it to the annotation.
        let annotated = self.annotated_locals();
        for (local, data) in body.locals.iter() {
            let slot = &mut local_tys[local.into_raw().into_u32() as usize];
            if slot.is_some() {
                continue;
            }
            let ty = data.ty.as_ref();
            *slot = Some(if annotated.contains(&local) && self.ctx.closure_local.is_none() {
                self.written(ty)
            } else {
                self.renumber(ty)
            });
        }
        self.regions.local_tys = local_tys.into_iter().map(|it| it.unwrap()).collect();
    }

    /// What a `let` with a type written out requires, as rustc's user type annotations: that
    /// the type is well-formed, and for a `_` pattern that the value has the type.
    fn annotations(&mut self) {
        let mut lets = Vec::new();
        for (_, expr) in self.ctx.store.exprs() {
            let hir_def::hir::Expr::Block { statements, .. } = expr else { continue };
            for statement in statements.iter() {
                if let Statement::Let { pat, type_ref: Some(_), initializer, .. } = statement {
                    lets.push((*pat, *initializer));
                }
            }
        }
        for (pat, initializer) in lets {
            let Some(ty) = self.ctx.infer.type_of_pat.get(pat).map(|it| it.as_ref()) else {
                continue;
            };
            let ty = self.written(ty);
            let at = At { category: Category::TypeAnnotation, span: MirSpan::PatId(pat) };
            self.well_formed(ty, at, 0);
            if let Pat::Wild = self.ctx.store[pat]
                && let Some(initializer) = initializer
                && let Some(local) = self.path_local(initializer)
            {
                let value = self.regions.local_tys[local.into_raw().into_u32() as usize];
                self.relate(value, ty, Variance::Covariant, at);
            }
        }
    }

    /// The local of the variable `expr` names, if it is a path to one of this body's.
    fn path_local(&self, expr: hir_def::hir::ExprId) -> Option<LocalId> {
        use hir_def::resolver::{HasResolver, ValueNs};
        let hir_def::hir::Expr::Path(path) = &self.ctx.store[expr] else { return None };
        let db = self.ctx.db;
        let owner = self.body.owner;
        let mut resolver = owner.resolver(db);
        let _guard = resolver.update_to_inner_scope(db, owner.expression_store_owner(db), expr);
        let hygiene = self.ctx.store.expr_path_hygiene(expr);
        match resolver.resolve_path_in_value_ns_fully(db, path, hygiene)? {
            ValueNs::LocalBinding(binding) => self.body.binding_locals.get(binding).copied(),
            _ => None,
        }
    }

    /// The constraints the type `ty` holds being well-formed: in `&'a T` that `T: 'a`, and
    /// the where clauses of the structs and enums it names.
    fn well_formed(&mut self, ty: Ty<'db>, at: At, depth: usize) {
        if depth > 8 {
            return;
        }
        match ty.kind() {
            TyKind::Ref(region, inner, _) => {
                self.type_outlives(inner, region, at);
                self.well_formed(inner, at, depth + 1);
            }
            TyKind::Adt(def, args) => {
                self.predicates(GenericDefId::AdtId(def.def_id()), args, at, true);
                for arg in args.iter() {
                    if let GenericArgKind::Type(it) = arg.kind() {
                        self.well_formed(it, at, depth + 1);
                    }
                }
            }
            TyKind::Tuple(tys) => {
                for it in tys.iter() {
                    self.well_formed(it, at, depth + 1);
                }
            }
            TyKind::Array(inner, _) | TyKind::Slice(inner) | TyKind::RawPtr(inner, _) => {
                self.well_formed(inner, at, depth + 1)
            }
            _ => (),
        }
    }

    fn annotated_locals(&self) -> FxHashSet<LocalId> {
        let mut found = FxHashSet::default();
        for (_, expr) in self.ctx.store.exprs() {
            let hir_def::hir::Expr::Block { statements, .. } = expr else { continue };
            for statement in statements.iter() {
                if let Statement::Let { pat, type_ref: Some(_), .. } = statement
                    && let Pat::Bind { id, subpat: None } = self.ctx.store[*pat]
                    && let Some(&local) = self.body.binding_locals.get(id)
                {
                    found.insert(local);
                }
            }
        }
        found
    }

    /// What the signature says of which universal regions outlive which: its where clauses
    /// and what its parameters' types imply, as `&'a T` that `T: 'a`.
    fn known_relations(&mut self) {
        let generic_def: Option<GenericDefId> = match self.body.owner {
            InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(it)) => Some(it.into()),
            InferBodyId::DefWithBodyId(DefWithBodyId::ConstId(it)) => Some(it.into()),
            InferBodyId::DefWithBodyId(DefWithBodyId::StaticId(it)) => Some(it.into()),
            _ => None,
        };
        let mut pairs = Vec::new();
        if let Some(def) = generic_def
            && self.ctx.closure_local.is_none()
        {
            for clause in GenericPredicates::query_all(self.ctx.db, def).skip_binder() {
                match clause.kind().skip_binder() {
                    ClauseKind::RegionOutlives(OutlivesPredicate(longer, shorter)) => {
                        if let (Some(longer), Some(shorter)) = (
                            self.signature_region(longer, rustc_type_ir::INNERMOST),
                            self.signature_region(shorter, rustc_type_ir::INNERMOST),
                        ) && let (Some(longer), Some(shorter)) = (rv_of(longer), rv_of(shorter))
                        {
                            pairs.push((longer, shorter));
                        }
                    }
                    ClauseKind::TypeOutlives(OutlivesPredicate(ty, region)) => {
                        if let Some(region) =
                            self.signature_region(region, rustc_type_ir::INNERMOST).and_then(rv_of)
                        {
                            self.regions.type_bounds.push((ty, region));
                        }
                    }
                    _ => (),
                }
            }
        }
        // The bounds the where clauses imply too, as of supertraits: `T: Trait` where
        // `trait Trait: 'static`.
        if self.ctx.closure_local.is_none() {
            let clauses: Vec<_> = self.ctx.env.caller_bounds().iter().collect();
            for clause in clauses {
                match clause.kind().skip_binder() {
                    ClauseKind::TypeOutlives(OutlivesPredicate(ty, region)) => {
                        if let Some(region) =
                            self.signature_region(region, rustc_type_ir::INNERMOST).and_then(rv_of)
                        {
                            self.regions.type_bounds.push((ty, region));
                        }
                    }
                    ClauseKind::RegionOutlives(OutlivesPredicate(longer, shorter)) => {
                        if let (Some(longer), Some(shorter)) = (
                            self.signature_region(longer, rustc_type_ir::INNERMOST),
                            self.signature_region(shorter, rustc_type_ir::INNERMOST),
                        ) && let (Some(longer), Some(shorter)) = (rv_of(longer), rv_of(shorter))
                        {
                            pairs.push((longer, shorter));
                        }
                    }
                    _ => (),
                }
            }
        }
        let signature: Vec<Ty<'db>> = self
            .body
            .param_locals
            .iter()
            .chain((!self.returns_opaque).then_some(&super::return_slot()))
            .map(|&it| self.regions.local_tys[it.into_raw().into_u32() as usize])
            .collect();
        let mut signature = signature;
        signature.extend(self.impl_header_types());
        let mut seen = FxHashSet::default();
        for ty in signature {
            self.implied_bounds(ty, &mut pairs, &mut seen, 0);
        }
        let count = self.regions.universals.len();
        let mut known = vec![BitSet::new(count); count];
        let static_index = self.regions.universal(self.regions.static_rv).unwrap();
        for (index, row) in known.iter_mut().enumerate() {
            row.insert(index);
            // Every universal region outlives the body.
            row.insert(1);
        }
        for shorter in 0..count {
            known[static_index].insert(shorter);
        }
        for (longer, shorter) in pairs {
            if let (Some(longer), Some(shorter)) =
                (self.regions.universal(longer), self.regions.universal(shorter))
            {
                known[longer].insert(shorter);
            }
        }
        // What is known is closed under transitivity.
        for middle in 0..count {
            for longer in 0..count {
                if known[longer].contains(middle) {
                    let row = known[middle].clone();
                    known[longer].union(&row);
                }
            }
        }
        self.regions.known = known;
    }

    /// The self type of the impl `function` is in with its anonymous lifetimes universal
    /// regions, when it has any.
    fn impl_self_of(&mut self, function: hir_def::FunctionId) -> Option<(Ty<'db>, Ty<'db>)> {
        let ItemContainerId::ImplId(impl_) = function.lookup(self.ctx.db).container else {
            return None;
        };
        let raw = self.ctx.db.impl_self_ty(impl_).instantiate_identity().skip_norm_wip();
        let mut anonymous = false;
        let named =
            rustc_type_ir::fold_regions(self.interner(), raw, |region, _| match region.kind() {
                RegionKind::ReError(_) => {
                    anonymous = true;
                    self.new_universal(UniversalKind::AnonParam)
                }
                _ => region,
            });
        if !anonymous {
            return None;
        }
        self.impl_self = Some((raw, named));
        self.impl_self
    }

    /// The self type and trait arguments of the impl a method is in, whose well-formedness the
    /// method assumes as it does its parameters'.
    fn impl_header_types(&mut self) -> Vec<Ty<'db>> {
        let InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(function)) = self.body.owner
        else {
            return Vec::new();
        };
        if self.ctx.closure_local.is_some() {
            return Vec::new();
        }
        let ItemContainerId::ImplId(impl_) = function.lookup(self.ctx.db).container else {
            return Vec::new();
        };
        let db = self.ctx.db;
        let raw = db.impl_self_ty(impl_).instantiate_identity().skip_norm_wip();
        let mut tys = vec![match self.impl_self {
            Some((from, to)) if from == raw => to,
            _ => raw,
        }];
        if let Some(trait_ref) = db.impl_trait(impl_) {
            let trait_ref = trait_ref.instantiate_identity().skip_norm_wip();
            tys.extend(trait_ref.args.iter().skip(1).filter_map(|arg| match arg.kind() {
                GenericArgKind::Type(it) => Some(it),
                _ => None,
            }));
        }
        tys.into_iter()
            .map(|ty| {
                rustc_type_ir::fold_regions(self.interner(), ty, |region, depth| {
                    self.signature_region(region, depth).unwrap_or_else(|| self.fresh())
                })
            })
            .collect()
    }

    fn implied_bounds(
        &mut self,
        ty: Ty<'db>,
        pairs: &mut Vec<(Rv, Rv)>,
        seen: &mut FxHashSet<Ty<'db>>,
        depth: usize,
    ) {
        if depth > 4 || !seen.insert(ty) {
            return;
        }
        match ty.kind() {
            TyKind::Ref(region, inner, _) => {
                if let Some(shorter) = rv_of(region) {
                    let referent = match inner.kind() {
                        TyKind::Alias(_) => self.normalize(inner).unwrap_or(inner),
                        _ => inner,
                    };
                    let mut components = Vec::new();
                    let mut params = Vec::new();
                    outlives_components(referent, &mut components, &mut params);
                    pairs.extend(components.into_iter().map(|longer| (longer, shorter)));
                    self.regions.type_bounds.extend(params.into_iter().map(|it| (it, shorter)));
                }
                self.implied_bounds(inner, pairs, seen, depth + 1);
            }
            TyKind::Adt(def, args) => {
                for arg in args.iter() {
                    if let GenericArgKind::Type(arg) = arg.kind() {
                        self.implied_bounds(arg, pairs, seen, depth + 1);
                    }
                }
                // What the fields imply is what rustc infers of the type's parameters.
                let adt = def.def_id();
                for variant in adt_variants(self.ctx.db, adt) {
                    for (_, field) in self.ctx.db.field_types(variant).iter() {
                        let field = field.ty().instantiate(self.interner(), args).skip_norm_wip();
                        self.implied_bounds(field, pairs, seen, depth + 1);
                    }
                }
                // And the bounds the type declares, which its well-formedness needs.
                let clauses: Vec<_> = GenericPredicates::query_all(self.ctx.db, adt.into())
                    .iter_instantiated(self.interner(), args.as_slice())
                    .map(|it| it.skip_norm_wip())
                    .collect();
                for clause in clauses {
                    match clause.kind().skip_binder() {
                        ClauseKind::RegionOutlives(OutlivesPredicate(longer, shorter)) => {
                            if let (Some(longer), Some(shorter)) =
                                (self.region_rv(longer), self.region_rv(shorter))
                            {
                                pairs.push((longer, shorter));
                            }
                        }
                        ClauseKind::TypeOutlives(OutlivesPredicate(ty, region)) => {
                            if let Some(region) = self.region_rv(region) {
                                self.regions.type_bounds.push((ty, region));
                            }
                        }
                        _ => (),
                    }
                }
            }
            TyKind::Tuple(tys) => {
                for it in tys.iter() {
                    self.implied_bounds(it, pairs, seen, depth + 1);
                }
            }
            TyKind::Array(inner, _) | TyKind::Slice(inner) | TyKind::RawPtr(inner, _) => {
                self.implied_bounds(inner, pairs, seen, depth + 1)
            }
            TyKind::Alias(_) => {
                if let Some(normalized) = self.normalize(ty)
                    && normalized != ty
                {
                    self.implied_bounds(normalized, pairs, seen, depth + 1);
                }
            }
            _ => (),
        }
    }

    /// `ty` an associated type normalized, with the lifetimes it takes from its arguments
    /// the same regions: the solver gives the result regions of its own, related to those it
    /// came from only by constraints it records.
    fn normalize(&mut self, ty: Ty<'db>) -> Option<Ty<'db>> {
        use hir_ty::next_solver::{
            infer::{region_constraints::Constraint, traits::ObligationCause},
            obligation_ctxt::ObligationCtxt,
        };
        let infcx = &self.ctx.infcx;
        let constraints_before = infcx
            .inner
            .borrow_mut()
            .unwrap_region_constraints()
            .region_constraint_data()
            .constraints
            .len();
        let first_new = infcx.num_region_vars();
        let mut ocx = ObligationCtxt::new(infcx);
        let normalized =
            ocx.structurally_normalize_ty(&ObligationCause::dummy(), self.ctx.env, ty).ok()?;
        let mut same = FxHashMap::default();
        {
            let mut inner = infcx.inner.borrow_mut();
            let mut collector = inner.unwrap_region_constraints();
            let data = collector.region_constraint_data();
            for constraint in &data.constraints[constraints_before..] {
                if let Constraint::VarSubVar(a, b) = *constraint {
                    let (new, old) = if a.index() >= first_new { (a, b) } else { (b, a) };
                    if new.index() >= first_new && old.index() < first_new {
                        same.entry(new).or_insert(old);
                    }
                }
            }
        }
        let mut unmapped = false;
        let normalized = rustc_type_ir::fold_regions(self.interner(), normalized, |region, _| {
            match region.kind() {
                RegionKind::ReVar(vid) if vid.index() >= first_new => match same.get(&vid) {
                    Some(&old) => Region::new_var(self.interner(), old),
                    None => {
                        unmapped = true;
                        region
                    }
                },
                _ => region,
            }
        });
        (!unmapped).then_some(normalized)
    }

    fn outlives(&mut self, sup: Rv, sub: Rv, at: At) {
        if sup != sub {
            let location = self.location;
            self.regions.outlives.push(Outlives {
                sup,
                sub,
                location,
                category: at.category,
                span: at.span,
            });
        }
    }

    fn relate_regions(&mut self, a: Region<'db>, b: Region<'db>, variance: Variance, at: At) {
        let (Some(a), Some(b)) = (rv_of(a), rv_of(b)) else { return };
        match variance {
            Variance::Covariant => self.outlives(a, b, at),
            Variance::Contravariant => self.outlives(b, a, at),
            Variance::Invariant => {
                self.outlives(a, b, at);
                self.outlives(b, a, at);
            }
            Variance::Bivariant => (),
        }
    }

    /// That `a` is a subtype of `b` under `variance`.
    fn relate(&mut self, a: Ty<'db>, b: Ty<'db>, variance: Variance, at: At) {
        if a == b || variance == Variance::Bivariant {
            return;
        }
        match (a.kind(), b.kind()) {
            (TyKind::Ref(ra, ta, ma), TyKind::Ref(rb, tb, _)) => {
                self.relate_regions(ra, rb, variance, at);
                let inner = match ma {
                    rustc_ast_ir::Mutability::Not => variance,
                    rustc_ast_ir::Mutability::Mut => variance.xform(Variance::Invariant),
                };
                self.relate(ta, tb, inner, at);
            }
            (TyKind::RawPtr(ta, ma), TyKind::RawPtr(tb, _)) => {
                let inner = match ma {
                    rustc_ast_ir::Mutability::Not => variance,
                    rustc_ast_ir::Mutability::Mut => variance.xform(Variance::Invariant),
                };
                self.relate(ta, tb, inner, at);
            }
            (TyKind::Adt(da, aa), TyKind::Adt(db, ab)) if da == db => {
                let variances = self.ctx.db.variances_of(GenericDefId::AdtId(da.def_id()));
                self.relate_args(aa, ab, Some(variances.as_slice()), variance, at);
            }
            (TyKind::Tuple(xs), TyKind::Tuple(ys)) => {
                for (x, y) in xs.iter().zip(ys.iter()) {
                    self.relate(x, y, variance, at);
                }
            }
            (TyKind::Array(x, _), TyKind::Array(y, _)) | (TyKind::Slice(x), TyKind::Slice(y)) => {
                self.relate(x, y, variance, at)
            }
            (TyKind::FnPtr(sa, _), TyKind::FnPtr(sb, _)) => {
                let (xs, ys) =
                    (sa.skip_binder().inputs_and_output, sb.skip_binder().inputs_and_output);
                let count = xs.len().min(ys.len());
                for (index, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
                    let inner = if index + 1 == count {
                        variance
                    } else {
                        variance.xform(Variance::Contravariant)
                    };
                    self.relate(x, y, inner, at);
                }
            }
            (TyKind::Dynamic(pa, ra), TyKind::Dynamic(pb, rb)) => {
                self.relate_regions(ra, rb, variance, at);
                for (x, y) in pa.iter().zip(pb.iter()) {
                    let (x, y) = (x.skip_binder(), y.skip_binder());
                    if let (
                        rustc_type_ir::ExistentialPredicate::Trait(x),
                        rustc_type_ir::ExistentialPredicate::Trait(y),
                    ) = (x, y)
                    {
                        self.relate_args(x.args, y.args, None, Variance::Invariant, at);
                    }
                }
            }
            (TyKind::Alias(x), TyKind::Alias(y)) if x.kind == y.kind => {
                self.relate_args(x.args, y.args, None, Variance::Invariant, at)
            }
            (TyKind::Closure(_, aa), TyKind::Closure(_, ab))
            | (TyKind::FnDef(_, aa), TyKind::FnDef(_, ab)) => {
                self.relate_args(aa, ab, None, Variance::Invariant, at)
            }
            _ => (),
        }
    }

    fn relate_args(
        &mut self,
        a: GenericArgs<'db>,
        b: GenericArgs<'db>,
        variances: Option<&[Variance]>,
        variance: Variance,
        at: At,
    ) {
        for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let param =
                variances.and_then(|it| it.get(index).copied()).unwrap_or(Variance::Invariant);
            let inner = variance.xform(param);
            match (x.kind(), y.kind()) {
                (GenericArgKind::Lifetime(x), GenericArgKind::Lifetime(y)) => {
                    self.relate_regions(x, y, inner, at)
                }
                (GenericArgKind::Type(x), GenericArgKind::Type(y)) => self.relate(x, y, inner, at),
                _ => (),
            }
        }
    }

    /// That `ty` outlives `region`: each region in it does, and each type parameter in it
    /// must, which is a type test.
    fn type_outlives(&mut self, ty: Ty<'db>, region: Region<'db>, at: At) {
        let Some(region) = rv_of(region) else { return };
        let mut components = Vec::new();
        let mut params = Vec::new();
        outlives_components(ty, &mut components, &mut params);
        for longer in components {
            self.outlives(longer, region, at);
        }
        for param in params {
            self.regions.type_tests.push(TypeTest { ty: param, lower: region, span: at.span });
        }
    }

    fn place_ty(&self, local: LocalId, projection: &[PlaceElem]) -> PlaceTy<'db> {
        PlaceTy::from_ty(self.regions.local_tys[local.into_raw().into_u32() as usize])
            .multi_projection_ty(&self.ctx.infcx, self.ctx.env, projection)
    }

    fn operand_ty(&self, operand: &Operand) -> Option<Ty<'db>> {
        match &operand.kind {
            OperandKind::Copy(place) | OperandKind::Move(place) => {
                Some(self.place_ty(place.local, place.projection.as_slice()).ty)
            }
            OperandKind::Constant { ty, .. } => Some(self.renumber(ty.as_ref())),
            OperandKind::Allocation { allocation } => Some(self.renumber(allocation.as_ref().ty)),
            OperandKind::Static(_) => None,
        }
    }

    /// `value` with the lifetimes a user wrote, the signature's, kept and the others fresh.
    fn written<T: rustc_type_ir::TypeFoldable<hir_ty::next_solver::DbInterner<'db>>>(
        &mut self,
        value: T,
    ) -> T {
        let keep_static = writes_static_object(self.ctx.store);
        let value = value.fold_with(&mut ObjectDefaults {
            db: self.ctx.db,
            interner: self.interner(),
            keep_static,
            erase: true,
        });
        rustc_type_ir::fold_regions(self.interner(), value, |region, depth| match region.kind() {
            RegionKind::ReEarlyParam(_) | RegionKind::ReLateParam(_) | RegionKind::ReStatic => {
                self.signature_region(region, depth).unwrap_or_else(|| self.fresh())
            }
            RegionKind::ReBound(..) => region,
            _ => self.fresh(),
        })
    }

    /// The type of a callee whose path spells out generic arguments, `f::<&'a T>`, with the
    /// lifetimes written there: rustc's user type annotation on the path.
    fn written_callee(&mut self, func: &Operand, call: MirSpan) -> Option<Ty<'db>> {
        let ty = match &func.kind {
            OperandKind::Constant { ty, .. } => ty.as_ref(),
            OperandKind::Allocation { allocation } => allocation.as_ref().ty,
            _ => return None,
        };
        let expr = match (func.span(), call) {
            (Some(MirSpan::ExprId(expr)), _) => expr,
            (_, MirSpan::ExprId(call)) => match self.ctx.store[call] {
                hir_def::hir::Expr::Call { callee, .. } => callee,
                _ => return None,
            },
            _ => return None,
        };
        let hir_def::hir::Expr::Path(path) = &self.ctx.store[expr] else { return None };
        if !spells_generics(path) {
            return None;
        }
        Some(self.written(ty))
    }

    fn category_of_assignment(&self, local: LocalId, projection: &[PlaceElem]) -> Category {
        if !projection.is_empty() {
            return Category::Assignment;
        }
        if local == super::return_slot() {
            Category::Return
        } else if self.ctx.bindings.contains_key(&local) {
            Category::Assignment
        } else {
            Category::Boring
        }
    }

    fn statement(&mut self, kind: &StatementKind, span: MirSpan) {
        let StatementKind::Assign(place, rvalue) = kind else { return };
        let projection = place.projection.as_slice();
        let dest = self.place_ty(place.local, projection).ty;
        let at = At { category: self.category_of_assignment(place.local, projection), span };
        if let MirSpan::ExprId(expr) = span
            && let hir_def::hir::Expr::Cast { .. } = self.ctx.store[expr]
            && matches!(rvalue, Rvalue::Use(_) | Rvalue::Cast(..))
            && let Some(written) = self.ctx.infer.type_of_expr.get(expr)
        {
            let written = self.written(written.as_ref());
            let at = At { category: Category::TypeAnnotation, span };
            self.relate(dest, written, Variance::Invariant, at);
        }
        match rvalue {
            Rvalue::Repeat(operand, _) => {
                if let (Some(ty), TyKind::Array(element, _)) =
                    (self.operand_ty(operand), dest.kind())
                {
                    self.relate(ty, element, Variance::Covariant, at);
                }
            }
            Rvalue::Aggregate(AggregateKind::Array(_), operands) => {
                if let TyKind::Array(element, _) = dest.kind() {
                    for operand in operands.iter() {
                        if let Some(ty) = self.operand_ty(operand) {
                            let at = At { span: operand.span().unwrap_or(span), ..at };
                            self.relate(ty, element, Variance::Covariant, at);
                        }
                    }
                }
            }
            Rvalue::Aggregate(AggregateKind::Union(_, field), operands) => {
                if let (TyKind::Adt(_, args), Some(operand)) = (dest.kind(), operands.first())
                    && let Some(ty) = self.operand_ty(operand)
                {
                    let field_ty = self.ctx.db.field_types(field.parent)[field.local_id]
                        .ty()
                        .instantiate(self.interner(), args)
                        .skip_norm_wip();
                    self.relate(ty, field_ty, Variance::Covariant, at);
                }
            }
            _ => {
                if let Some(ty) = self.rvalue_ty(rvalue, span) {
                    self.relate(ty, dest, Variance::Covariant, at);
                }
            }
        }
    }

    fn rvalue_ty(&mut self, rvalue: &Rvalue, span: MirSpan) -> Option<Ty<'db>> {
        let boring = At { category: Category::Boring, span };
        match rvalue {
            Rvalue::Use(operand) => self.operand_ty(operand),
            Rvalue::CopyForDeref(place) => {
                Some(self.place_ty(place.local, place.projection.as_slice()).ty)
            }
            Rvalue::Ref(kind, place) => {
                let region = self.fresh();
                let rv = rv_of(region).unwrap();
                if let Some(location) = self.location {
                    self.regions.loan_regions.insert(location, rv);
                }
                let projection = place.projection.as_slice();
                self.reborrow_constraints(place.local, projection, rv, boring);
                let ty = self.place_ty(place.local, projection).ty;
                let mutability = match kind {
                    BorrowKind::Mut { .. } => rustc_ast_ir::Mutability::Mut,
                    _ => rustc_ast_ir::Mutability::Not,
                };
                Some(Ty::new_ref(self.interner(), region, ty, mutability))
            }
            Rvalue::Cast(kind, operand, target) => {
                let target = self.renumber(target.as_ref());
                let source = self.operand_ty(operand)?;
                let at = At { category: Category::Cast, span };
                match kind {
                    CastKind::PointerCoercion(hir_ty::PointerCast::Unsize) => {
                        self.unsize(source, target, at)
                    }
                    CastKind::PointerCoercion(hir_ty::PointerCast::MutToConstPointer) => {
                        if let (TyKind::RawPtr(x, _), TyKind::RawPtr(y, _)) =
                            (source.kind(), target.kind())
                        {
                            self.relate(x, y, Variance::Covariant, at);
                        }
                    }
                    _ => (),
                }
                Some(target)
            }
            Rvalue::Aggregate(AggregateKind::Adt(variant, args), operands) => {
                let args = match span {
                    MirSpan::ExprId(expr)
                        if matches!(&self.ctx.store[expr],
                            hir_def::hir::Expr::RecordLit { path, .. } if spells_generics(path)) =>
                    {
                        self.written(args.as_ref())
                    }
                    _ => self.renumber(args.as_ref()),
                };
                let fields = self.ctx.db.field_types(*variant);
                for ((_, field), operand) in fields.iter().zip(operands.iter()) {
                    let field = field.ty().instantiate(self.interner(), args).skip_norm_wip();
                    if let Some(ty) = self.operand_ty(operand) {
                        let at = At { span: operand.span().unwrap_or(span), ..boring };
                        self.relate(ty, field, Variance::Covariant, at);
                    }
                }
                let adt = match variant {
                    hir_def::VariantId::EnumVariantId(it) => {
                        AdtId::EnumId(it.lookup(self.ctx.db).parent)
                    }
                    hir_def::VariantId::StructId(it) => AdtId::StructId(*it),
                    hir_def::VariantId::UnionId(it) => AdtId::UnionId(*it),
                };
                Some(Ty::new_adt(self.interner(), adt, args))
            }
            Rvalue::Aggregate(AggregateKind::Tuple(ty), operands) => {
                let ty = self.renumber(ty.as_ref());
                if let TyKind::Tuple(fields) = ty.kind() {
                    for (field, operand) in fields.iter().zip(operands.iter()) {
                        if let Some(it) = self.operand_ty(operand) {
                            self.relate(it, field, Variance::Covariant, boring);
                        }
                    }
                }
                Some(ty)
            }
            Rvalue::Aggregate(AggregateKind::Closure(ty), operands) => {
                let ty = self.renumber(ty.as_ref());
                if let TyKind::Closure(_, args) = ty.kind() {
                    let upvars = args.as_closure().tupled_upvars_ty();
                    if let TyKind::Tuple(upvars) = upvars.kind() {
                        for (upvar, operand) in upvars.iter().zip(operands.iter()) {
                            if let Some(it) = self.operand_ty(operand) {
                                let at = At {
                                    category: Category::Usage,
                                    span: operand.span().unwrap_or(span),
                                };
                                self.relate(it, upvar, Variance::Covariant, at);
                            }
                        }
                    }
                }
                Some(ty)
            }
            _ => None,
        }
    }

    /// rustc's `add_reborrow_constraint`: the loan of `*p` outlives nothing longer than the
    /// reference `p` is, up to the first shared reference.
    fn reborrow_constraints(&mut self, local: LocalId, projection: &[PlaceElem], loan: Rv, at: At) {
        for index in (0..projection.len()).rev() {
            if projection[index] != ProjectionElem::Deref {
                continue;
            }
            let base = self.place_ty(local, &projection[..index]).ty;
            match base.kind() {
                TyKind::Ref(region, _, mutability) => {
                    if let Some(region) = rv_of(region) {
                        self.outlives(region, loan, at);
                    }
                    if mutability == rustc_ast_ir::Mutability::Not {
                        break;
                    }
                }
                TyKind::RawPtr(..) => break,
                _ => (),
            }
        }
    }

    fn unsize(&mut self, source: Ty<'db>, target: Ty<'db>, at: At) {
        match (source.kind(), target.kind()) {
            (TyKind::Ref(ra, ta, _), TyKind::Ref(rb, tb, _)) => {
                self.relate_regions(ra, rb, Variance::Covariant, at);
                self.unsize_pointee(ta, tb, at);
            }
            (TyKind::RawPtr(ta, _), TyKind::RawPtr(tb, _)) => self.unsize_pointee(ta, tb, at),
            (TyKind::Adt(da, aa), TyKind::Adt(db, ab)) if da == db => {
                for (x, y) in aa.iter().zip(ab.iter()) {
                    match (x.kind(), y.kind()) {
                        (GenericArgKind::Type(x), GenericArgKind::Type(y)) => {
                            if matches!(x.kind(), TyKind::Adt(..))
                                && matches!(y.kind(), TyKind::Adt(..))
                            {
                                self.unsize(x, y, at);
                            } else {
                                self.unsize_pointee(x, y, at);
                            }
                        }
                        (GenericArgKind::Lifetime(x), GenericArgKind::Lifetime(y)) => {
                            self.relate_regions(x, y, Variance::Covariant, at)
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        }
    }

    fn unsize_pointee(&mut self, source: Ty<'db>, target: Ty<'db>, at: At) {
        match (source.kind(), target.kind()) {
            (TyKind::Array(x, _), TyKind::Slice(y)) => self.relate(x, y, Variance::Covariant, at),
            (TyKind::Dynamic(..), TyKind::Dynamic(..)) => {
                self.relate(source, target, Variance::Covariant, at)
            }
            (_, TyKind::Dynamic(_, region)) => self.type_outlives(source, region, at),
            _ => self.relate(source, target, Variance::Covariant, at),
        }
    }

    fn terminator(&mut self, kind: &TerminatorKind, span: MirSpan) {
        match kind {
            TerminatorKind::Call { func, args, destination, .. } => {
                let Some(func_ty) =
                    self.written_callee(func, span).or_else(|| self.operand_ty(func))
                else {
                    return;
                };
                let (inputs, output) = match func_ty.kind() {
                    TyKind::FnDef(def, fn_args) => {
                        let def: CallableDefId = def.0;
                        let sig = self
                            .ctx
                            .db
                            .callable_item_signature(def)
                            .instantiate(self.interner(), fn_args)
                            .skip_norm_wip();
                        let sig = self.ctx.infcx.instantiate_binder_with_fresh_vars(
                            hir_ty::Span::Dummy,
                            hir_ty::next_solver::infer::BoundRegionConversionTime::FnCall,
                            sig,
                        );
                        self.callee_predicates(def, fn_args, span);
                        let (self_param, keep_static) = match def {
                            CallableDefId::FunctionId(it) => {
                                let written = FunctionSignature::of(self.ctx.db, it);
                                (written.has_self_param(), writes_static_object(&written.store))
                            }
                            _ => (false, true),
                        };
                        let (inputs, output) = self.elide(
                            sig.inputs().iter().copied().collect(),
                            sig.output(),
                            self_param,
                            keep_static,
                        );
                        // The where clauses of the associated types in the signature hold
                        // for it to be well-formed.
                        let at = At { category: Category::Predicate, span };
                        for ty in inputs.iter().copied().chain([output]) {
                            for (def, args) in projections(ty) {
                                if let Ok(def) = GenericDefId::try_from(def) {
                                    self.predicates(def, args, at, false);
                                }
                            }
                        }
                        (inputs, output)
                    }
                    TyKind::FnPtr(sig_tys, header) => {
                        let sig = sig_tys.with(header);
                        let sig = self.ctx.infcx.instantiate_binder_with_fresh_vars(
                            hir_ty::Span::Dummy,
                            hir_ty::next_solver::infer::BoundRegionConversionTime::FnCall,
                            sig,
                        );
                        (sig.inputs().iter().copied().collect::<Vec<_>>(), sig.output())
                    }
                    _ => return,
                };
                for (arg, input) in args.iter().zip(inputs) {
                    if let Some(ty) = self.operand_ty(arg) {
                        let at = At {
                            category: Category::CallArgument,
                            span: arg.span().unwrap_or(span),
                        };
                        self.relate(ty, input, Variance::Covariant, at);
                    }
                }
                let projection = destination.projection.as_slice();
                let dest = self.place_ty(destination.local, projection).ty;
                let at = At {
                    category: self.category_of_assignment(destination.local, projection),
                    span,
                };
                self.relate(output, dest, Variance::Covariant, at);
            }
            TerminatorKind::DropAndReplace { place, value, .. } => {
                let projection = place.projection.as_slice();
                let dest = self.place_ty(place.local, projection).ty;
                if let Some(ty) = self.operand_ty(value) {
                    let at =
                        At { category: self.category_of_assignment(place.local, projection), span };
                    self.relate(ty, dest, Variance::Covariant, at);
                }
            }
            _ => (),
        }
    }

    /// A callee's signature with the lifetimes it elides, which RA leaves as errors, made
    /// region variables by the elision rules: each elided input lifetime its own, an elided
    /// output lifetime that of `self` when `self` is a reference, else that of the only
    /// lifetime of the inputs. The default lifetime of a trait object is not one of them.
    fn elide(
        &self,
        inputs: Vec<Ty<'db>>,
        output: Ty<'db>,
        self_param: bool,
        keep_static: bool,
    ) -> (Vec<Ty<'db>>, Ty<'db>) {
        let interner = self.interner();
        let mut defaults = ObjectDefaults { db: self.ctx.db, interner, keep_static, erase: false };
        let inputs: Vec<Ty<'db>> = inputs
            .into_iter()
            .map(|ty| {
                let ty =
                    rustc_type_ir::fold_regions(interner, ty, |region, _| match region.kind() {
                        RegionKind::ReError(_) => self.fresh(),
                        _ => region,
                    });
                rustc_type_ir::TypeFoldable::fold_with(ty, &mut defaults)
            })
            .collect();
        let self_region = match inputs.first().map(|it| it.kind()) {
            Some(TyKind::Ref(region, ..)) if self_param => Some(region),
            _ => None,
        };
        let elided = self_region.or_else(|| {
            let mut distinct = Vec::new();
            for ty in &inputs {
                rustc_type_ir::fold_regions(interner, *ty, |region, _| {
                    if !matches!(region.kind(), RegionKind::ReBound(..))
                        && !distinct.contains(&region)
                    {
                        distinct.push(region);
                    }
                    region
                });
            }
            match distinct.as_slice() {
                [only] => Some(*only),
                _ => None,
            }
        });
        let output =
            rustc_type_ir::fold_regions(interner, output, |region, _| match region.kind() {
                RegionKind::ReError(_) => elided.unwrap_or_else(|| self.fresh()),
                _ => region,
            });
        (inputs, rustc_type_ir::TypeFoldable::fold_with(output, &mut defaults))
    }

    /// The where clauses of what a call calls, which the caller must prove.
    fn callee_predicates(&mut self, def: CallableDefId, args: GenericArgs<'db>, span: MirSpan) {
        let generic_def: GenericDefId = match def {
            CallableDefId::FunctionId(it) => it.into(),
            CallableDefId::StructId(it) => it.into(),
            CallableDefId::EnumVariantId(it) => it.lookup(self.ctx.db).parent.into(),
        };
        let at = At { category: Category::Predicate, span };
        self.predicates(generic_def, args, at, true);
    }

    /// The where clauses of `def` with `args`, which hold where it is used. Without
    /// `type_outlives`, only those between regions.
    fn predicates(
        &mut self,
        def: GenericDefId,
        args: GenericArgs<'db>,
        at: At,
        type_outlives: bool,
    ) {
        let clauses: Vec<_> = GenericPredicates::query_all(self.ctx.db, def)
            .iter_instantiated(self.interner(), args.as_slice())
            .map(|it| it.skip_norm_wip())
            .collect();
        for clause in clauses {
            match clause.kind().skip_binder() {
                ClauseKind::RegionOutlives(OutlivesPredicate(longer, shorter)) => {
                    if let (Some(longer), Some(shorter)) =
                        (self.region_rv(longer), self.region_rv(shorter))
                    {
                        self.outlives(longer, shorter, at);
                    }
                }
                ClauseKind::TypeOutlives(OutlivesPredicate(ty, region)) if type_outlives => {
                    let region = match region.kind() {
                        RegionKind::ReStatic => self.static_region(),
                        _ => region,
                    };
                    self.type_outlives(ty, region, at);
                }
                ClauseKind::Trait(predicate) if type_outlives => {
                    let trait_ref = predicate.trait_ref;
                    for region in self.trait_outlives(trait_ref) {
                        self.type_outlives(trait_ref.self_ty(), region, at);
                    }
                    self.caller_bound(trait_ref, at)
                }
                _ => (),
            }
        }
    }

    /// The regions a type implementing `trait_ref` outlives, by the trait's `Self: 'r` bounds:
    /// its own, instantiated, and `'static` from its supertraits.
    fn trait_outlives(
        &mut self,
        trait_ref: rustc_type_ir::TraitRef<hir_ty::next_solver::DbInterner<'db>>,
    ) -> Vec<Region<'db>> {
        let db = self.ctx.db;
        let mut regions = Vec::new();
        if static_trait(db, trait_ref.def_id.0) {
            regions.push(self.static_region());
        }
        for clause in GenericPredicates::query_all(db, trait_ref.def_id.0.into())
            .iter_instantiated(self.interner(), trait_ref.args.as_slice())
        {
            if let ClauseKind::TypeOutlives(OutlivesPredicate(ty, region)) =
                clause.skip_norm_wip().kind().skip_binder()
                && ty == trait_ref.self_ty()
                && !matches!(region.kind(), RegionKind::ReStatic | RegionKind::ReBound(..))
            {
                regions.push(region);
            }
        }
        regions
    }

    /// A trait bound on a type parameter, proven by the one where clause of this body that
    /// names the same trait for it, whose arguments it then equals.
    fn caller_bound(
        &mut self,
        trait_ref: rustc_type_ir::TraitRef<hir_ty::next_solver::DbInterner<'db>>,
        at: At,
    ) {
        let TyKind::Param(param) = trait_ref.self_ty().kind() else { return };
        let mut found = None;
        for clause in self.ctx.env.caller_bounds().iter() {
            let kind = clause.kind();
            let ClauseKind::Trait(bound) = kind.skip_binder() else { continue };
            if bound.trait_ref.def_id != trait_ref.def_id
                || !matches!(bound.trait_ref.self_ty().kind(), TyKind::Param(it) if it.index == param.index)
            {
                continue;
            }
            if found.is_some() || !kind.bound_vars().is_empty() {
                return;
            }
            found = Some(bound.trait_ref);
        }
        let Some(bound) = found else { return };
        let args = rustc_type_ir::fold_regions(self.interner(), bound.args, |region, depth| {
            self.signature_region(region, depth).unwrap_or_else(|| self.fresh())
        });
        self.relate_args(trait_ref.args, args, None, Variance::Invariant, at);
    }

    fn region_rv(&self, region: Region<'db>) -> Option<Rv> {
        match region.kind() {
            RegionKind::ReStatic => Some(self.regions.static_rv),
            _ => rv_of(region),
        }
    }
}

/// The lifetime of a trait object written without one, which rust-analyzer gives as
/// `'static`: in `&'r dyn Trait` it is `'r`, and elsewhere in a function body inferred, unless
/// the body writes `dyn Trait + 'static` somewhere, where it may be that, or the trait's
/// bounds say `'static`, as rustc's `compute_object_lifetime_bound`.
struct ObjectDefaults<'db> {
    db: &'db dyn hir_ty::db::HirDatabase,
    interner: hir_ty::next_solver::DbInterner<'db>,
    keep_static: bool,
    /// In a body, where the lifetime is inferred rather than `'static`.
    erase: bool,
}

impl<'db> rustc_type_ir::TypeFolder<hir_ty::next_solver::DbInterner<'db>> for ObjectDefaults<'db> {
    fn cx(&self) -> hir_ty::next_solver::DbInterner<'db> {
        self.interner
    }

    fn fold_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
        use rustc_type_ir::{
            TypeFoldable, TypeSuperFoldable, inherent::BoundExistentialPredicates as _,
        };
        let is_static = |region: Region<'db>| matches!(region.kind(), RegionKind::ReStatic);
        let defaulted = |this: &Self, predicates: <hir_ty::next_solver::DbInterner<'db> as rustc_type_ir::Interner>::BoundExistentialPredicates, object| {
            is_static(object)
                && !this.keep_static
                && !predicates.principal_def_id().is_some_and(|it| static_trait(this.db, it.0))
        };
        match ty.kind() {
            TyKind::Ref(region, inner, mutability) if !is_static(region) => {
                if let TyKind::Dynamic(predicates, object) = inner.kind()
                    && defaulted(self, predicates, object)
                {
                    let inner = Ty::new_dynamic(self.interner, predicates.fold_with(self), region);
                    return Ty::new_ref(self.interner, region, inner, mutability);
                }
                ty.super_fold_with(self)
            }
            TyKind::Dynamic(predicates, object)
                if self.erase && defaulted(self, predicates, object) =>
            {
                Ty::new_dynamic(
                    self.interner,
                    predicates.fold_with(self),
                    Region::new_erased(self.interner),
                )
            }
            _ => ty.super_fold_with(self),
        }
    }
}

/// Each occurrence of an impl's self type made the one with universal regions for its
/// anonymous lifetimes.
struct ImplSelf<'db> {
    interner: hir_ty::next_solver::DbInterner<'db>,
    impl_self: (Ty<'db>, Ty<'db>),
}

impl<'db> rustc_type_ir::TypeFolder<hir_ty::next_solver::DbInterner<'db>> for ImplSelf<'db> {
    fn cx(&self) -> hir_ty::next_solver::DbInterner<'db> {
        self.interner
    }

    fn fold_ty(&mut self, ty: Ty<'db>) -> Ty<'db> {
        use rustc_type_ir::TypeSuperFoldable;
        if ty == self.impl_self.0 { self.impl_self.1 } else { ty.super_fold_with(self) }
    }
}

/// Whether `trait_` or one of its supertraits is bounded by `'static`, so that its object
/// type's lifetime defaults to that and each type implementing it outlives it.
fn static_trait(db: &dyn hir_ty::db::HirDatabase, trait_: hir_def::TraitId) -> bool {
    hir_ty::utils::all_super_traits(db, trait_).iter().any(|&it| {
        GenericPredicates::query_all(db, it.into()).skip_binder().any(|clause| {
            matches!(clause.kind().skip_binder(),
                ClauseKind::TypeOutlives(OutlivesPredicate(ty, region))
                    if matches!(ty.kind(), TyKind::Param(param) if param.index == 0)
                        && matches!(region.kind(), RegionKind::ReStatic))
        })
    })
}

/// Whether the type `ty` in `store` writes `Self`.
fn writes_self(
    store: &hir_def::expr_store::ExpressionStore,
    ty: hir_def::hir::type_ref::TypeRefId,
) -> bool {
    use hir_def::{expr_store::path::GenericArg, hir::type_ref::TypeRef};
    match &store[ty] {
        TypeRef::Path(path) => {
            path.is_self_type()
                || path.type_anchor().is_some_and(|it| writes_self(store, it))
                || path.generic_args().into_iter().flatten().flatten().any(|args| {
                    args.args
                        .iter()
                        .any(|arg| matches!(arg, GenericArg::Type(it) if writes_self(store, *it)))
                })
        }
        TypeRef::Tuple(tys) => tys.iter().any(|&it| writes_self(store, it)),
        TypeRef::RawPtr(it, _) | TypeRef::Slice(it) => writes_self(store, *it),
        TypeRef::Reference(it) => writes_self(store, it.ty),
        TypeRef::Array(it) => writes_self(store, it.ty),
        _ => false,
    }
}

/// Whether a type in `store` writes `dyn Trait + 'static`.
fn writes_static_object(store: &hir_def::expr_store::ExpressionStore) -> bool {
    use hir_def::hir::type_ref::{LifetimeRef, TypeBound, TypeRef};
    store.types.iter().any(|(_, it)| {
        match it {
        TypeRef::DynTrait(bounds) => bounds.iter().any(|bound| {
            matches!(bound, TypeBound::Lifetime(it) if matches!(store[*it], LifetimeRef::Static))
        }),
        _ => false,
    }
    })
}

/// Whether `path` writes generic arguments, `f::<T>` or `<T as Trait>::f`: a user type
/// annotation, whose lifetimes bind.
fn spells_generics(path: &hir_def::expr_store::path::Path) -> bool {
    path.type_anchor().is_some()
        || path.generic_args().is_some_and(|args| args.iter().any(Option::is_some))
}

fn adt_variants(db: &dyn hir_ty::db::HirDatabase, adt: AdtId) -> Vec<hir_def::VariantId> {
    match adt {
        AdtId::StructId(it) => vec![it.into()],
        AdtId::UnionId(it) => vec![it.into()],
        AdtId::EnumId(it) => {
            it.enum_variants(db).variants.values().map(|&(it, _)| it.into()).collect()
        }
    }
}

/// The regions and the type parameters in `ty` that must outlive a region for `ty` to: rustc's
/// outlives components.
pub(super) fn outlives_components<'db>(
    ty: Ty<'db>,
    regions: &mut Vec<Rv>,
    params: &mut Vec<Ty<'db>>,
) {
    match ty.kind() {
        TyKind::Param(_) => params.push(ty),
        TyKind::Alias(..) | TyKind::FnPtr(..) | TyKind::Placeholder(_) | TyKind::Bound(..) => (),
        TyKind::Ref(region, inner, _) => {
            regions.extend(rv_of(region));
            outlives_components(inner, regions, params);
        }
        TyKind::Dynamic(predicates, region) => {
            regions.extend(rv_of(region));
            for predicate in predicates.iter() {
                if let rustc_type_ir::ExistentialPredicate::Trait(it) = predicate.skip_binder() {
                    args_components(it.args, regions, params);
                }
            }
        }
        TyKind::Adt(_, args) | TyKind::FnDef(_, args) => args_components(args, regions, params),
        TyKind::Closure(_, args) => {
            outlives_components(args.as_closure().tupled_upvars_ty(), regions, params)
        }
        TyKind::Tuple(tys) => {
            for it in tys.iter() {
                outlives_components(it, regions, params);
            }
        }
        TyKind::Array(inner, _) | TyKind::Slice(inner) | TyKind::RawPtr(inner, _) => {
            outlives_components(inner, regions, params)
        }
        _ => (),
    }
}

fn args_components<'db>(args: GenericArgs<'db>, regions: &mut Vec<Rv>, params: &mut Vec<Ty<'db>>) {
    for arg in args.iter() {
        match arg.kind() {
            GenericArgKind::Lifetime(region) => regions.extend(rv_of(region)),
            GenericArgKind::Type(ty) => outlives_components(ty, regions, params),
            GenericArgKind::Const(_) => (),
        }
    }
}

/// The points each region holds at, and past them the universal regions it outlives.
pub(super) struct RegionValues {
    pub(super) points: usize,
    pub(super) values: Vec<BitSet>,
}

impl RegionValues {
    pub(super) fn contains_point(&self, rv: Rv, point: usize) -> bool {
        self.values[rv].contains(point)
    }

    pub(super) fn universals_of(&self, rv: Rv) -> impl Iterator<Item = usize> + '_ {
        let points = self.points;
        self.values[rv].iter().filter(move |&it| it >= points).map(move |it| it - points)
    }
}

/// Region inference: the values liveness gives grown along the outlives constraints until
/// each region holds wherever a region it outlives does.
pub(super) fn infer(
    regions: &Regions<'_>,
    count: usize,
    points: usize,
    mut values: Vec<BitSet>,
) -> RegionValues {
    let size = points + regions.universals.len();
    values.resize(count, BitSet::new(size));
    for (index, &(rv, _)) in regions.universals.iter().enumerate() {
        let value = &mut values[rv];
        for point in 0..points {
            value.insert(point);
        }
        value.insert(points + index);
    }
    let mut successors: Vec<Vec<Rv>> = vec![Vec::new(); count];
    for constraint in &regions.outlives {
        if constraint.sup < count && constraint.sub < count {
            successors[constraint.sub].push(constraint.sup);
        }
    }
    let mut queued = vec![true; count];
    let mut work: Vec<Rv> = (0..count).collect();
    while let Some(sub) = work.pop() {
        queued[sub] = false;
        if successors[sub].is_empty() {
            continue;
        }
        let value = values[sub].clone();
        for &sup in &successors[sub] {
            if values[sup].union(&value) && !queued[sup] {
                queued[sup] = true;
                work.push(sup);
            }
        }
    }
    RegionValues { points, values }
}

/// The constraints from `from` down to `to`, the shortest way, for the error that blames one
/// of them.
pub(super) fn constraint_path(
    regions: &Regions<'_>,
    count: usize,
    from: Rv,
    to: Rv,
) -> Vec<Outlives> {
    let mut by_sup: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (index, constraint) in regions.outlives.iter().enumerate() {
        if constraint.sup < count {
            by_sup[constraint.sup].push(index);
        }
    }
    let mut came_from: Vec<Option<usize>> = vec![None; count];
    let mut seen = vec![false; count];
    let mut queue = std::collections::VecDeque::from([from]);
    seen[from] = true;
    while let Some(rv) = queue.pop_front() {
        if rv == to {
            let mut path = Vec::new();
            let mut at = to;
            while let Some(index) = came_from[at] {
                path.push(regions.outlives[index]);
                at = regions.outlives[index].sup;
            }
            path.reverse();
            return path;
        }
        for &index in &by_sup[rv] {
            let sub = regions.outlives[index].sub;
            if !seen[sub] {
                seen[sub] = true;
                came_from[sub] = Some(index);
                queue.push_back(sub);
            }
        }
    }
    Vec::new()
}

/// The constraint an error about `path` points at, as rustc's `best_blame_constraint`.
pub(super) fn blame(path: &[Outlives]) -> Option<Outlives> {
    let best = path.iter().map(|it| it.category).min()?;
    path.iter()
        .find(|it| it.category == best && it.span != MirSpan::Unknown)
        .or_else(|| path.iter().find(|it| it.category == best))
        .copied()
}
