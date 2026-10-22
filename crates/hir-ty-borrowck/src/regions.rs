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
    resolver::HasResolver as _,
    signatures::FunctionSignature,
};
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{
    ClauseKind, OutlivesPredicate, RegionKind, Variance,
    inherent::{GenericArgs as _, IntoKind, ParamEnv as _, Region as _, SliceLike as _, Ty as _},
};

use hir_ty::{
    GenericPredicates, InferBodyId,
    db::GeneralConstId,
    mir::{
        AggregateKind, BorrowKind, CastKind, LocalId, MirBody, MirSpan, Operand, OperandKind,
        PlaceElem, PlaceTy, ProjectionElem, Rvalue, StatementKind, TerminatorKind,
    },
    next_solver::{ConstKind, GenericArgKind, GenericArgs, Region, Ty, TyKind},
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
    /// A closure's requirement, proven where the closure is made.
    pub(super) from_closure: bool,
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
    /// In a closure, a region of the closure's type: one of the body that creates it, which
    /// rustc's `UniversalRegions` calls external. The closure's own, the late-bound lifetimes
    /// of its signature and the borrow of its environment, are `AnonParam`.
    Creator,
}

/// The position among a closure's creator's regions of the enclosing item's late-bound
/// lifetime 0, past those of the closure's type.
pub(super) const LATE_PARENT: usize = 1 << 24;

/// What a closure needs of its creator's regions that it cannot prove itself, as rustc's
/// `ClosureOutlivesRequirement`: that the region at position `longer` among the regions of
/// the closure's type outlives the one at `shorter`, or `'static` for `None`. With a `subject`,
/// that type parameter outlives it instead, a type test the creator proves.
#[derive(Clone, Copy, Debug)]
pub(super) struct Requirement<'db> {
    pub(super) subject: Option<Ty<'db>>,
    pub(super) longer: usize,
    pub(super) shorter: Option<usize>,
    pub(super) category: Category,
    pub(super) span: MirSpan,
}

/// That the type parameter `ty` must outlive `lower`, as rustc's `TypeTest`.
pub(super) struct TypeTest<'db> {
    pub(super) ty: Ty<'db>,
    pub(super) lower: Rv,
    pub(super) span: MirSpan,
    /// From a `for<'a> T: 'a` bound, which rustc reports without a code.
    pub(super) forall: bool,
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
    /// Whether universal regions are checked against each other: not in a closure whose
    /// signature could not be split into its own regions and its creator's.
    pub(super) check_universals: bool,
    /// The body is a method of a trait impl, whose signature the trait fixes: rustc's
    /// `is_bound_region_in_impl_item`, which keeps E0621 from suggesting a change to it.
    pub(super) is_impl_item: bool,
    /// The universal regions of the return type (an `async fn`'s declared one) and of `self`:
    /// rustc's `is_return_type_anon` and `is_self_anon`, for which E0621 is not given either.
    pub(super) unsuggested: FxHashSet<usize>,
    /// In a closure called by reference, the universal region of that reference.
    pub(super) closure_env: Option<usize>,
    /// In a closure, the position among the regions of its type of each of its creator's
    /// universal regions, by universal region.
    pub(super) creator_positions: FxHashMap<usize, usize>,
    /// The regions of a returned `impl Trait`'s hidden type, each of which must be one of
    /// `choices`, `'static` and the lifetimes the `impl Trait` captures: rustc's member
    /// constraints. Empty when what it captures is not known exactly.
    pub(super) members: Vec<Rv>,
    pub(super) choices: Vec<Rv>,
    /// The body returns an `impl Trait` that captures no lifetime, of a hidden type that names
    /// no other alias: what it returns must be `'static`, rustc's `OpaqueType` constraint.
    pub(super) returns_static_opaque: bool,
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

/// Which arguments of the `impl Trait` `def`, of `len` of them, its hidden type may name: the
/// others are bivariant in rustc's `variances_of` the opaque type, and its liveness skips
/// them. Every one, unless `use<..>` lists them; before edition 2024, outside a trait and its
/// impls and outside an `async fn`, the type parameters and the lifetimes its bounds name.
/// `None` where this does not work it out, which counts every argument captured.
pub(super) fn opaque_captures(
    db: &dyn hir_ty::db::HirDatabase,
    def: hir_ty::next_solver::SolverDefId<'_>,
    len: usize,
) -> Option<Vec<bool>> {
    use hir_def::{
        GenericParamId,
        hir::{
            generics::GenericParamDataRef,
            type_ref::{LifetimeRef, TypeBound, TypeRef, UseArgRef},
        },
    };
    use rustc_type_ir::Interner as _;
    let hir_ty::next_solver::SolverDefId::InternedOpaqueTyId(opaque) = def else { return None };
    let hir_ty::ImplTraitId::ReturnTypeImplTrait(function, index) = opaque.loc(db) else {
        return None;
    };
    let params: Vec<_> = hir_ty::generics::generics(db, function.into()).iter(true).collect();
    if params.len() != len {
        return None;
    }
    let is_lifetime = |id: &GenericParamId| matches!(id, GenericParamId::LifetimeParamId(_));
    let signature = FunctionSignature::of(db, function);
    let written = signature.ret_type.and_then(|ret| match &signature.store[ret] {
        TypeRef::ImplTrait(bounds) if index.into_raw().into_u32() == 0 => Some(bounds),
        _ => None,
    });
    let listed = written.and_then(|bounds| {
        bounds.iter().find_map(|bound| match bound {
            TypeBound::Use(args) => Some(args),
            _ => None,
        })
    });
    let mut named = FxHashSet::default();
    let mut unknown = false;
    let interner = hir_ty::next_solver::DbInterner::new_no_crate(db);
    let mut without_self = hir_ty::next_solver::util::BottomUpFolder {
        interner,
        ty_op: |ty: Ty<'_>| match ty.kind() {
            TyKind::Alias(alias) if alias.kind.def_id() == def => Ty::new_unit(interner),
            _ => ty,
        },
        lt_op: |region| region,
        ct_op: |ct| ct,
    };
    for clause in interner.item_bounds(def).skip_binder() {
        let clause = rustc_type_ir::TypeFoldable::fold_with(clause, &mut without_self);
        rustc_type_ir::fold_regions(interner, clause, |region, _| {
            match region.kind() {
                RegionKind::ReEarlyParam(it) => {
                    named.insert(it.index as usize);
                }
                RegionKind::ReStatic | RegionKind::ReBound(..) => (),
                _ => unknown = true,
            }
            region
        });
    }
    if let Some(listed) = listed {
        let captured = params
            .iter()
            .map(|(id, data)| {
                let (
                    GenericParamId::LifetimeParamId(param),
                    GenericParamDataRef::LifetimeParamData(data),
                ) = (id, data)
                else {
                    return true;
                };
                listed.iter().any(|arg| match arg {
                    UseArgRef::Lifetime(it) => match &signature.store[*it] {
                        LifetimeRef::Named(name) => *name == data.name,
                        LifetimeRef::Param(it) => it == param,
                        _ => false,
                    },
                    UseArgRef::Name(_) => false,
                })
            })
            .collect::<Vec<_>>();
        // A lifetime the bounds name and `use<..>` leaves out is an error before borrow
        // checking.
        let bounds_listed =
            named.iter().all(|&index| captured.get(index).copied().unwrap_or(false));
        return (!unknown && bounds_listed).then_some(captured);
    }
    if unknown {
        return None;
    }
    let in_trait = match function.lookup(db).container {
        ItemContainerId::TraitId(_) => true,
        ItemContainerId::ImplId(it) => db.impl_trait(it).is_some(),
        _ => false,
    };
    let edition = hir_def::HasModule::krate(&function, db).data(db).edition;
    if edition.at_least_2024() || in_trait || signature.is_async() {
        return None;
    }
    Some(
        params
            .iter()
            .enumerate()
            .map(|(index, (id, _))| !is_lifetime(id) || named.contains(&index))
            .collect(),
    )
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

/// Whether `ty` has a closure or a coroutine in it, whose signature RA does not keep
/// higher-ranked: `|line: &String|` is `fn(&'1 String)` for one `'1`, not `for<'a>`.
fn names_closure(ty: Ty<'_>) -> bool {
    ty.walk().filter_map(|it| rustc_type_ir::inherent::GenericArg::as_type(&it)).any(|it| {
        matches!(
            it.kind(),
            TyKind::Closure(..) | TyKind::CoroutineClosure(..) | TyKind::Coroutine(..)
        )
    })
}

/// The regions of a hidden type that rustc's `CollectMemberConstraintsVisitor` gives member
/// constraints: not those of a closure's parent arguments, nor an `impl Trait`'s arguments
/// it does not capture.
fn member_regions<'db>(
    db: &dyn hir_ty::db::HirDatabase,
    ty: Ty<'db>,
    aliases: bool,
    out: &mut Vec<Rv>,
) {
    struct Collect<'o, 'd> {
        db: &'d dyn hir_ty::db::HirDatabase,
        aliases: bool,
        out: &'o mut Vec<Rv>,
    }
    impl Collect<'_, '_> {
        fn args<'db>(&mut self, args: GenericArgs<'db>, skip: usize) {
            for arg in args.iter().skip(skip) {
                rustc_type_ir::TypeVisitable::visit_with(&arg, self);
            }
        }
    }
    impl<'db> rustc_type_ir::TypeVisitor<hir_ty::next_solver::DbInterner<'db>> for Collect<'_, '_> {
        type Result = ();
        fn visit_region(&mut self, region: Region<'db>) {
            if let Some(rv) = rv_of(region) {
                self.out.push(rv);
            }
        }
        fn visit_ty(&mut self, ty: Ty<'db>) {
            match ty.kind() {
                TyKind::Closure(_, args) => self.args(args, args.as_closure().parent_args().len()),
                TyKind::Coroutine(_, args) => {
                    self.args(args, args.as_coroutine().parent_args().len())
                }
                TyKind::CoroutineClosure(_, args) => {
                    self.args(args, args.as_coroutine_closure().parent_args().len())
                }
                TyKind::Alias(_) if !self.aliases => (),
                TyKind::Alias(alias)
                    if let rustc_type_ir::AliasTyKind::Opaque { .. } = alias.kind
                        && let Some(captured) =
                            opaque_captures(self.db, alias.kind.def_id(), alias.args.len()) =>
                {
                    for (arg, captured) in alias.args.iter().zip(captured) {
                        if captured {
                            rustc_type_ir::TypeVisitable::visit_with(&arg, self);
                        }
                    }
                }
                _ => rustc_type_ir::TypeSuperVisitable::super_visit_with(&ty, self),
            }
        }
    }
    rustc_type_ir::TypeVisitable::visit_with(&ty, &mut Collect { db, aliases, out });
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

pub(super) fn build<'db>(
    ctx: &Ctx<'_, 'db>,
    body: &MirBody<'db>,
    cfg: &Cfg,
    requirements: &FxHashMap<hir_def::hir::ExprId, (usize, Vec<Requirement<'db>>)>,
) -> Regions<'db> {
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
            unsuggested: FxHashSet::default(),
            closure_env: None,
            creator_positions: FxHashMap::default(),
            members: Vec::new(),
            returns_static_opaque: false,
            choices: Vec::new(),
        },
        requirements,
        early: FxHashMap::default(),
        late: FxHashMap::default(),
        parent_late: FxHashMap::default(),
        impl_self: None,
        returns_opaque: false,
        elided_output: None,
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
    /// What each closure checked so far needs of the body that creates it, with how many
    /// regions the closure's type has.
    requirements: &'c FxHashMap<hir_def::hir::ExprId, (usize, Vec<Requirement<'db>>)>,
    /// The universal region of each lifetime parameter, by its index.
    early: FxHashMap<u32, Region<'db>>,
    /// The universal region of each of the signature's late-bound lifetimes.
    late: FxHashMap<u32, Region<'db>>,
    /// In a closure, the universal region of each late-bound lifetime of the enclosing item
    /// it names, one of its creator's.
    parent_late: FxHashMap<u32, Region<'db>>,
    /// The self type of the impl the body is in, as rust-analyzer gives it with `'_` an error,
    /// and with universal regions there: rustc's anonymous lifetime parameters of the impl.
    impl_self: Option<(Ty<'db>, Ty<'db>)>,
    /// The return type is an `impl Trait`, so the return slot has the type it hides, which
    /// implies nothing to the body.
    returns_opaque: bool,
    /// The region lifetime elision gives the elided lifetimes of the return type.
    elided_output: Option<Region<'db>>,
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
        // A closure's enclosing item's lifetime parameters are its creator's.
        let kind = match self.ctx.closure_local {
            Some(_) => UniversalKind::Creator,
            None => UniversalKind::Named,
        };
        let region = self.new_universal(kind);
        self.early.insert(index, region);
        self.regions.early.insert(index, rv_of(region).unwrap());
        region
    }

    /// The universal region of the enclosing item's late-bound lifetime `var`: the signature's
    /// in the item, a creator's region of a closure's, at the position [`LATE_PARENT`] + `var`,
    /// as rustc's `for_each_late_bound_region_in_recursive_scope` makes them external.
    fn parent_late_region(&mut self, var: u32) -> Region<'db> {
        if self.ctx.closure_local.is_none() {
            if let Some(&it) = self.late.get(&var) {
                return it;
            }
            let region = self.new_universal(UniversalKind::Named);
            self.late.insert(var, region);
            return region;
        }
        if let Some(&it) = self.parent_late.get(&var) {
            return it;
        }
        let region = self.new_universal(UniversalKind::Creator);
        self.regions
            .creator_positions
            .insert(self.regions.universals.len() - 1, LATE_PARENT + var as usize);
        self.parent_late.insert(var, region);
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
            // In a closure, a late-bound lifetime of the enclosing item is its creator's.
            RegionKind::ReLateParam(late) if self.ctx.closure_local.is_some() => {
                Some(self.parent_late_region(late.bound_region.var.as_u32()))
            }
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
            self.regions.check_universals = self.closure_signature(&mut local_tys);
        }
        if self.ctx.closure_local.is_some() && !self.regions.check_universals {
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
        } else if self.ctx.closure_local.is_none() {
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
                        let ty = local_tys[it.into_raw().into_u32() as usize]?;
                        self.self_reference(function, ty).or(match ty.kind() {
                            TyKind::Ref(region, ..) => Some(region),
                            _ => None,
                        })
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
            self.elided_output = elided;
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
            let captures_lifetime = signature.as_ref().is_some_and(|(_, output)| {
                let TyKind::Alias(alias) = output.kind() else { return true };
                let Some(captured) =
                    opaque_captures(self.ctx.db, alias.kind.def_id(), alias.args.len())
                else {
                    return true;
                };
                alias.args.iter().zip(captured).any(|(arg, captured)| {
                    captured && matches!(arg.kind(), GenericArgKind::Lifetime(_))
                })
            });
            self.regions.returns_static_opaque = hidden_static
                && !captures_lifetime
                && !rustc_type_ir::TypeVisitableExt::has_aliases(&ty);
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
            let mut unsuggested = Vec::new();
            regions_of(ty, &mut unsuggested);
            if let InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(function)) = owner
                && FunctionSignature::of(self.ctx.db, function).has_self_param()
                && let Some(&local) = body.param_locals.first()
                && let Some(ty) = local_tys[local.into_raw().into_u32() as usize]
            {
                regions_of(ty, &mut unsuggested);
            }
            for rv in unsuggested {
                self.regions.unsuggested.extend(self.regions.universal(rv));
            }
            if opaque
                && !hidden_static
                && let Some((_, output)) = &signature
                && let TyKind::Alias(alias) = output.kind()
                && let rustc_type_ir::AliasTyKind::Opaque { .. } = alias.kind
                && let Some(captured) =
                    opaque_captures(self.ctx.db, alias.kind.def_id(), alias.args.len())
            {
                let mut choices = vec![self.regions.static_rv];
                let mut known = true;
                for (arg, captured) in alias.args.iter().zip(captured) {
                    if let GenericArgKind::Lifetime(region) = arg.kind()
                        && captured
                    {
                        match self
                            .signature_region(region, rustc_type_ir::INNERMOST)
                            .and_then(rv_of)
                        {
                            Some(rv) => choices.push(rv),
                            None => known = false,
                        }
                    }
                }
                // What the bounds say the `impl Trait` outlives its hidden type must: rustc
                // proves them of it. A projection bound that names a lifetime constrains it in
                // ways not followed here, so its member constraints are left out.
                let mut outlived = Vec::new();
                let bounds =
                    rustc_type_ir::Interner::item_bounds(self.interner(), alias.kind.def_id());
                for clause in bounds.iter_instantiated(self.interner(), alias.args.as_slice()) {
                    match clause.kind().skip_binder() {
                        ClauseKind::TypeOutlives(OutlivesPredicate(bounded, region))
                            if let TyKind::Alias(bounded) = bounded.kind()
                                && bounded.kind.def_id() == alias.kind.def_id() =>
                        {
                            // Under the clause's own binder, the signature's is one further out.
                            let binder = rustc_type_ir::INNERMOST.shifted_in(1);
                            match self.signature_region(region, binder).and_then(rv_of) {
                                Some(rv) => outlived.push(rv),
                                None => known = false,
                            }
                        }
                        ClauseKind::Projection(projection)
                            if rustc_type_ir::TypeVisitableExt::has_free_regions(
                                &projection.term,
                            ) =>
                        {
                            known = false;
                        }
                        _ => (),
                    }
                }
                let mut members = Vec::new();
                member_regions(self.ctx.db, ty, true, &mut members);
                members.retain(|&it| it != self.regions.static_rv);
                members.sort_unstable();
                members.dedup();
                // An alias's own bounds may already say it outlives them, so the regions within
                // one are left to the member constraints.
                let mut bounded = Vec::new();
                member_regions(self.ctx.db, ty, false, &mut bounded);
                bounded.retain(|&it| it != self.regions.static_rv);
                bounded.sort_unstable();
                bounded.dedup();
                for &member in &bounded {
                    for &sub in &outlived {
                        self.regions.outlives.push(Outlives {
                            sup: member,
                            sub,
                            location: None,
                            category: Category::Boring,
                            span: MirSpan::Unknown,
                            from_closure: false,
                        });
                    }
                }
                if known {
                    self.regions.members = members;
                    self.regions.choices = choices;
                }
            }
            if let InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(function)) = owner {
                self.regions.is_impl_item = matches!(
                    function.lookup(self.ctx.db).container,
                    ItemContainerId::ImplId(it) if self.ctx.db.impl_trait(it).is_some()
                );
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
            *slot =
                Some(if annotated.contains(&local) { self.written(ty) } else { self.renumber(ty) });
        }
        self.regions.local_tys = local_tys.into_iter().map(|it| it.unwrap()).collect();
    }

    /// The arguments of a closure's or coroutine's type with each region a universal region of
    /// its creator's, by position. They start with the `parents` generic arguments of the
    /// enclosing item, which rustc makes the item's own parameters where the closure is made.
    fn creator_args(&mut self, args: GenericArgs<'db>, parents: usize) -> GenericArgs<'db> {
        let args =
            rustc_type_ir::fold_regions(self.interner(), args, |region, _| match region.kind() {
                RegionKind::ReBound(..) => region,
                _ => {
                    let region = self.new_universal(UniversalKind::Creator);
                    let position = self.regions.creator_positions.len();
                    self.regions
                        .creator_positions
                        .insert(self.regions.universals.len() - 1, position);
                    region
                }
            });
        for (index, arg) in args.iter().take(parents).enumerate() {
            if let GenericArgKind::Lifetime(region) = arg.kind()
                && let Some(rv) = rv_of(region)
            {
                self.early.insert(index as u32, region);
                self.regions.early.insert(index as u32, rv);
            }
        }
        args
    }

    /// A closure's parameters, return slot and environment typed as rustc's `UniversalRegions`
    /// types them: each region of the closure's type a universal region of its creator's, and
    /// the late-bound lifetimes of its signature and the borrow of its environment its own.
    /// `false`, with nothing set, when the closure's type does not have the shape for it.
    fn closure_signature(&mut self, local_tys: &mut [Option<Ty<'db>>]) -> bool {
        let body = self.body;
        let Some((&env_local, params)) = body.param_locals.split_first() else { return false };
        let env = body.locals[env_local].ty.as_ref();
        let (closure, by_ref) = match env.kind() {
            TyKind::Ref(_, inner, mutability) => (inner, Some(mutability)),
            _ => (env, None),
        };
        let interner = self.interner();
        if let TyKind::Coroutine(def, args) = closure.kind() {
            // An `async` body is polled through `&mut` its future, with a `&mut Context` of the
            // caller's; it returns what its type says the future's output is.
            let args = self.creator_args(args, args.as_coroutine().parent_args().len());
            let mut coroutine = Ty::new_coroutine(interner, def, args);
            if let Some(mutability) = by_ref {
                let region = self.new_universal(UniversalKind::AnonParam);
                self.regions.closure_env = Some(self.regions.universals.len() - 1);
                coroutine = Ty::new_ref(interner, region, coroutine, mutability);
            }
            local_tys[env_local.into_raw().into_u32() as usize] = Some(coroutine);
            for &local in params {
                let ty = rustc_type_ir::fold_regions(
                    interner,
                    body.locals[local].ty.as_ref(),
                    |region, _| match region.kind() {
                        RegionKind::ReBound(..) => region,
                        _ => self.new_universal(UniversalKind::AnonParam),
                    },
                );
                local_tys[local.into_raw().into_u32() as usize] = Some(ty);
            }
            local_tys[super::return_slot().into_raw().into_u32() as usize] =
                Some(args.as_coroutine().return_ty());
            return true;
        }
        let TyKind::Closure(def, args) = closure.kind() else { return false };
        let sig = args.as_closure().sig();
        let &[tupled] = sig.skip_binder().inputs().as_slice() else { return false };
        let TyKind::Tuple(inputs) = tupled.kind() else { return false };
        if inputs.len() != params.len() {
            return false;
        }
        let args = self.creator_args(args, args.as_closure().parent_args().len());
        let mut closure = Ty::new_closure(interner, def, args);
        if let Some(mutability) = by_ref {
            let region = self.new_universal(UniversalKind::AnonParam);
            self.regions.closure_env = Some(self.regions.universals.len() - 1);
            closure = Ty::new_ref(interner, region, closure, mutability);
        }
        let mut late = FxHashMap::default();
        let sig = rustc_type_ir::fold_regions(
            interner,
            args.as_closure().sig().skip_binder(),
            |region, depth| match region.kind() {
                RegionKind::ReBound(rustc_type_ir::BoundVarIndexKind::Bound(debruijn), bound)
                    if debruijn == depth =>
                {
                    *late
                        .entry(bound.var.as_u32())
                        .or_insert_with(|| self.new_universal(UniversalKind::AnonParam))
                }
                _ => region,
            },
        );
        let &[tupled] = sig.inputs().as_slice() else { return false };
        let TyKind::Tuple(inputs) = tupled.kind() else { return false };
        local_tys[env_local.into_raw().into_u32() as usize] = Some(closure);
        for (&local, ty) in params.iter().zip(inputs.iter()) {
            local_tys[local.into_raw().into_u32() as usize] = Some(ty);
        }
        local_tys[super::return_slot().into_raw().into_u32() as usize] = Some(sig.output());
        true
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
        self.pattern_paths();
        for (pat, initializer) in lets {
            let Some(ty) = self.ctx.infer.type_of_pat.get(pat).map(|it| it.as_ref()) else {
                continue;
            };
            let ty = self.written(ty);
            let at = At { category: Category::TypeAnnotation, span: MirSpan::PatId(pat) };
            self.well_formed(ty, at, 0);
            if !matches!(self.ctx.store[pat], Pat::Bind { subpat: None, .. }) {
                self.annotated_bindings(pat, ty, at, true);
            }
            if !matches!(self.ctx.store[pat], Pat::Bind { .. })
                && let Some(initializer) = initializer
            {
                let value = match self.path_local(initializer) {
                    Some(local) => {
                        Some(self.regions.local_tys[local.into_raw().into_u32() as usize])
                    }
                    None => self
                        .ascribed_place(pat)
                        .map(|(local, projection)| self.place_ty(local, &projection).ty),
                };
                if let Some(value) = value {
                    self.relate(value, ty, Variance::Covariant, at);
                }
            }
        }
    }

    /// The variables a pattern `pat` with the type `ty` written binds have the parts of `ty`
    /// they bind, as rustc's user type projections: in `let (y, _): (&'static u32, u32)`,
    /// `y` is a `&'static u32`. Not through a binding by reference or a pattern that derefs
    /// what it matches by default binding modes.
    fn annotated_bindings(&mut self, pat: hir_def::hir::PatId, ty: Ty<'db>, at: At, top: bool) {
        if self.ctx.infer.pat_adjustments.contains_key(&pat) {
            return;
        }
        let interner = self.interner();
        match &self.ctx.store[pat] {
            Pat::Bind { id, subpat } => {
                if !top
                    && let Some(hir_ty::BindingMode(by_ref, _)) = self.ctx.infer.binding_mode(pat)
                    && let Some(&local) = self.body.binding_locals.get(*id)
                {
                    let local_ty = self.regions.local_tys[local.into_raw().into_u32() as usize];
                    // `ref y` borrows the part, which has the type.
                    match (by_ref, local_ty.kind()) {
                        (hir_ty::ByRef::No, _) => {
                            self.relate(local_ty, ty, Variance::Invariant, at)
                        }
                        (hir_ty::ByRef::Yes(_), TyKind::Ref(_, inner, _)) => {
                            self.relate(inner, ty, Variance::Invariant, at)
                        }
                        _ => (),
                    }
                }
                if let Some(subpat) = *subpat {
                    self.annotated_bindings(subpat, ty, at, false);
                }
            }
            Pat::Tuple { args, ellipsis: None } => {
                if let TyKind::Tuple(tys) = ty.kind()
                    && tys.len() == args.len()
                {
                    let args = args.clone();
                    for (pat, ty) in args.iter().zip(tys.iter()) {
                        self.annotated_bindings(*pat, ty, at, false);
                    }
                }
            }
            Pat::Ref { pat, .. } => {
                if let TyKind::Ref(_, inner, _) = ty.kind() {
                    self.annotated_bindings(*pat, inner, at, false);
                }
            }
            Pat::Record { args, ellipsis: _, .. } => {
                let TyKind::Adt(_, adt_args) = ty.kind() else { return };
                let Some(variant) = self.ctx.infer.variant_resolution_for_pat(pat) else { return };
                let fields = hir_def::signatures::VariantFields::of(self.ctx.db, variant);
                let types = self.ctx.db.field_types(variant);
                let parts: Vec<_> = args
                    .iter()
                    .filter_map(|it| {
                        let field = fields.field(&it.name)?;
                        Some((it.pat, types[field].ty().instantiate(interner, adt_args)))
                    })
                    .collect();
                for (pat, field) in parts {
                    let mut field = field.skip_norm_wip();
                    if rustc_type_ir::TypeVisitableExt::has_aliases(&field) {
                        field = self.normalize(field).unwrap_or(field);
                    }
                    self.annotated_bindings(pat, field, at, false);
                }
            }
            Pat::TupleStruct { args, ellipsis: None, .. } => {
                let TyKind::Adt(_, adt_args) = ty.kind() else { return };
                let Some(variant) = self.ctx.infer.variant_resolution_for_pat(pat) else { return };
                let types = self.ctx.db.field_types(variant);
                let parts: Vec<_> = args
                    .iter()
                    .zip(types.iter())
                    .map(|(pat, (_, field))| (*pat, field.ty().instantiate(interner, adt_args)))
                    .collect();
                for (pat, field) in parts {
                    let mut field = field.skip_norm_wip();
                    if rustc_type_ir::TypeVisitableExt::has_aliases(&field) {
                        field = self.normalize(field).unwrap_or(field);
                    }
                    self.annotated_bindings(pat, field, at, false);
                }
            }
            _ => (),
        }
    }

    /// What a struct, variant or constant pattern whose path spells out generic arguments
    /// requires of the value it matches, `let Foo::<'static> { .. } = foo;` that `foo` is a
    /// `Foo<'static>`, as rustc's user type annotation on the pattern. Only for the whole
    /// pattern of a `let` or a `match` arm, matching a variable or what the `let` reads.
    fn pattern_paths(&mut self) {
        let spelled = |pat: &Pat| match pat {
            Pat::Record { path, .. } | Pat::TupleStruct { path, .. } | Pat::Path(path) => {
                spells_generics(path)
            }
            _ => false,
        };
        let mut found = Vec::new();
        for (_, expr) in self.ctx.store.exprs() {
            match expr {
                hir_def::hir::Expr::Block { statements, .. } => {
                    for statement in statements.iter() {
                        if let Statement::Let { pat, type_ref: None, initializer, .. } = statement
                            && spelled(&self.ctx.store[*pat])
                        {
                            found.push((*pat, *initializer));
                        }
                    }
                }
                hir_def::hir::Expr::Match { expr, arms } => {
                    for arm in arms.iter() {
                        if spelled(&self.ctx.store[arm.pat]) {
                            found.push((arm.pat, Some(*expr)));
                        }
                    }
                }
                _ => (),
            }
        }
        for (pat, value) in found {
            let Some(ty) = self.ctx.infer.type_of_pat.get(pat).map(|it| it.as_ref()) else {
                continue;
            };
            let value = match value.and_then(|it| self.path_local(it)) {
                Some(local) => Some(self.regions.local_tys[local.into_raw().into_u32() as usize]),
                None => self
                    .ascribed_place(pat)
                    .map(|(local, projection)| self.place_ty(local, &projection).ty),
            };
            let Some(value) = value else { continue };
            let ty = self.written(ty);
            let at = At { category: Category::TypeAnnotation, span: MirSpan::PatId(pat) };
            if let Pat::Path(_) = self.ctx.store[pat] {
                // A constant's value is compared with `PartialEq<Self>`, both sides of a type
                // they are subtypes of: `x: &'x str` matches a `&'static str`.
                let common = self.renumber(value);
                self.relate(value, common, Variance::Covariant, at);
                self.relate(ty, common, Variance::Covariant, at);
            } else {
                self.relate(value, ty, Variance::Covariant, at);
            }
        }
    }

    /// The place the `let` of the pattern `pat` reads its value from: the fake read lowered at
    /// the pattern.
    fn ascribed_place(&self, pat: hir_def::hir::PatId) -> Option<(LocalId, Vec<PlaceElem>)> {
        self.body.basic_blocks.iter().find_map(|(_, block)| {
            block.statements.iter().find_map(|statement| match &statement.kind {
                StatementKind::FakeRead(place) if statement.span == MirSpan::PatId(pat) => {
                    let place = place.as_ref();
                    Some((place.local, place.projection.as_slice().to_vec()))
                }
                _ => None,
            })
        })
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

    /// What a path naming the associated item of `container` through a type needs, as
    /// rustc's user type annotation on it: its generic arguments well-formed, and the impl's
    /// self type and trait arguments, `<Evil>::f` with `struct Evil<'a, 'b: 'a>` that `'b: 'a`.
    fn path_well_formed(
        &mut self,
        expr: hir_def::hir::ExprId,
        container: ItemContainerId,
        args: GenericArgs<'db>,
        span: MirSpan,
    ) {
        let hir_def::hir::Expr::Path(path) = &self.ctx.store[expr] else { return };
        if !names_through_type(self.ctx.db, path, container) {
            return;
        }
        let at = At { category: Category::TypeAnnotation, span };
        let mut tys: Vec<Ty<'db>> = args.types().collect();
        if let ItemContainerId::ImplId(impl_) = container {
            let db = self.ctx.db;
            tys.push(db.impl_self_ty(impl_).instantiate(self.interner(), args).skip_norm_wip());
            if let Some(trait_ref) = db.impl_trait(impl_) {
                let trait_ref = trait_ref.instantiate(self.interner(), args).skip_norm_wip();
                tys.extend(trait_ref.args.types().skip(1));
            }
        }
        for ty in tys {
            self.well_formed(ty, at, 0);
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
    /// The region of the one `&Self` in the type of a method's `self` (`self: Box<&Self>`,
    /// `Pin<&mut Self>`), which rustc's lifetime elision gives the elided lifetimes of the
    /// return type.
    fn self_reference(&self, function: hir_def::FunctionId, ty: Ty<'db>) -> Option<Region<'db>> {
        let ItemContainerId::ImplId(impl_) = function.lookup(self.ctx.db).container else {
            return None;
        };
        let interner = self.interner();
        let erase = |ty: Ty<'db>| {
            let ty = rustc_type_ir::fold_regions(interner, ty, |_, _| Region::new_static(interner));
            super::liveness::normalize_erased(self.ctx, ty).unwrap_or(ty)
        };
        let self_ty = erase(self.ctx.db.impl_self_ty(impl_).instantiate_identity().skip_norm_wip());
        let mut found = Vec::new();
        let mut stack = vec![ty];
        while let Some(ty) = stack.pop() {
            match ty.kind() {
                TyKind::Ref(region, inner, _) if erase(inner) == self_ty => found.push(region),
                TyKind::Ref(_, inner, _) => stack.push(inner),
                TyKind::Adt(_, args) => {
                    stack.extend(args.iter().filter_map(|it| match it.kind() {
                        GenericArgKind::Type(ty) => Some(ty),
                        _ => None,
                    }))
                }
                _ => (),
            }
        }
        found.dedup();
        match found.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

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
        if self.ctx.closure_local.is_some() {
            return Vec::new();
        }
        let container = match self.body.owner {
            InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(it)) => {
                it.lookup(self.ctx.db).container
            }
            InferBodyId::DefWithBodyId(DefWithBodyId::ConstId(it)) => {
                it.lookup(self.ctx.db).container
            }
            _ => return Vec::new(),
        };
        let ItemContainerId::ImplId(impl_) = container else {
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
                            if let Some(shorter) = self.region_rv(region) {
                                let mut components = Vec::new();
                                let mut params = Vec::new();
                                outlives_components(ty, &mut components, &mut params);
                                pairs.extend(components.into_iter().map(|it| (it, shorter)));
                                self.regions
                                    .type_bounds
                                    .extend(params.into_iter().chain([ty]).map(|it| (it, shorter)));
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
            // `dyn Any + 'r` is well-formed only where `'r` outlives what the trait requires of
            // `Self`: `trait Any: 'static`.
            TyKind::Dynamic(predicates, region) => {
                let Some(shorter) = rv_of(region) else { return };
                for predicate in predicates.iter() {
                    let trait_ = match predicate.skip_binder() {
                        rustc_type_ir::ExistentialPredicate::Trait(it) => it.def_id.0,
                        rustc_type_ir::ExistentialPredicate::AutoTrait(it) => it.0,
                        rustc_type_ir::ExistentialPredicate::Projection(_) => continue,
                    };
                    let requires_static = GenericPredicates::query_all(self.ctx.db, trait_.into())
                        .iter_identity()
                        .any(|clause| {
                            matches!(
                                clause.skip_norm_wip().kind().skip_binder(),
                                ClauseKind::TypeOutlives(OutlivesPredicate(ty, region))
                                    if matches!(ty.kind(), TyKind::Param(param) if param.index == 0)
                                        && matches!(region.kind(), RegionKind::ReStatic)
                            )
                        });
                    if requires_static {
                        pairs.push((shorter, self.regions.static_rv));
                    }
                }
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
            let collector = inner.unwrap_region_constraints();
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
                from_closure: false,
            });
        }
    }

    /// Where an `async fn` makes the future of its body (`coroutine`), that the future's
    /// output is the declared return type: rustc proves the `Future<Output = ..>` bound of the
    /// `impl Future` the function returns of it. `typed` is the output as type inference
    /// gives it, with an elided lifetime an error, `output` the same renumbered.
    fn async_fn_output(
        &mut self,
        coroutine: hir_def::hir::ExprId,
        typed: Ty<'db>,
        output: Ty<'db>,
        span: MirSpan,
    ) {
        use hir_def::hir::{ClosureKind, CoroutineSource, Expr};
        if self.ctx.closure_local.is_some()
            || !matches!(
                self.ctx.store[coroutine],
                Expr::Closure {
                    closure_kind: ClosureKind::Coroutine { source: CoroutineSource::Fn, .. },
                    ..
                }
            )
        {
            return;
        }
        let InferBodyId::DefWithBodyId(DefWithBodyId::FunctionId(function)) = self.body.owner
        else {
            return;
        };
        let written = FunctionSignature::of(self.ctx.db, function);
        let mut defaults = ObjectDefaults {
            db: self.ctx.db,
            interner: self.interner(),
            keep_static: writes_static_object(&written.store),
            erase: false,
        };
        let typed = rustc_type_ir::TypeFoldable::fold_with(typed, &mut defaults);
        let elided = self.elided_output;
        let declared = rustc_type_ir::fold_regions(self.interner(), typed, |region, depth| {
            match region.kind() {
                RegionKind::ReError(_) => elided.unwrap_or_else(|| self.fresh()),
                RegionKind::ReVar(_) => self.fresh(),
                _ => self.signature_region(region, depth).unwrap_or_else(|| self.fresh()),
            }
        });
        let mut unsuggested = Vec::new();
        regions_of(declared, &mut unsuggested);
        for rv in unsuggested {
            self.regions.unsuggested.extend(self.regions.universal(rv));
        }
        let at = At { category: Category::Boring, span };
        self.relate(output, declared, Variance::Invariant, at);
    }

    /// rustc's `prove_closure_bounds` where the closure `ty` is made: its enclosing item's
    /// arguments are this body's parameters, and what the closure needs of this body's
    /// regions holds, blamed on what in the closure needs it.
    fn closure_requirements(
        &mut self,
        closure: hir_def::hir::ExprId,
        ty: Ty<'db>,
        parents: Vec<(u32, Region<'db>)>,
        span: MirSpan,
    ) {
        let at = At { category: Category::Boring, span };
        for (index, region) in parents {
            let own = self.early_region(index);
            self.relate_regions(region, own, Variance::Invariant, at);
        }
        let Some((count, requirements)) = self.requirements.get(&closure) else { return };
        let mut regions = Vec::new();
        regions_of(ty, &mut regions);
        if regions.len() != *count {
            return;
        }
        let at_position = |this: &mut Self, position: usize| match position {
            _ if position >= LATE_PARENT => {
                rv_of(this.parent_late_region((position - LATE_PARENT) as u32))
            }
            _ => regions.get(position).copied(),
        };
        for requirement in requirements {
            let sub = match requirement.shorter {
                Some(it) => match at_position(self, it) {
                    Some(it) => it,
                    None => continue,
                },
                None => self.regions.static_rv,
            };
            if let Some(ty) = requirement.subject {
                self.regions.type_tests.push(TypeTest {
                    ty,
                    lower: sub,
                    span: requirement.span,
                    forall: false,
                });
                continue;
            }
            let Some(sup) = at_position(self, requirement.longer) else { continue };
            if sup != sub {
                self.regions.outlives.push(Outlives {
                    sup,
                    sub,
                    location: self.location,
                    category: requirement.category,
                    span: requirement.span,
                    from_closure: true,
                });
            }
        }
    }

    fn relate_regions(&mut self, a: Region<'db>, b: Region<'db>, variance: Variance, at: At) {
        let static_rv = self.regions.static_rv;
        let rv = |region: Region<'db>| match region.kind() {
            RegionKind::ReStatic => Some(static_rv),
            _ => rv_of(region),
        };
        let (Some(a), Some(b)) = (rv(a), rv(b)) else { return };
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
                let variances = match x.kind {
                    rustc_type_ir::AliasTyKind::Opaque { .. } => {
                        opaque_captures(self.ctx.db, x.kind.def_id(), x.args.len()).map(|it| {
                            it.into_iter()
                                .map(|captured| match captured {
                                    true => Variance::Invariant,
                                    false => Variance::Bivariant,
                                })
                                .collect::<Vec<_>>()
                        })
                    }
                    _ => None,
                };
                self.relate_args(x.args, y.args, variances.as_deref(), Variance::Invariant, at)
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
            self.regions.type_tests.push(TypeTest {
                ty: param,
                lower: region,
                span: at.span,
                forall: false,
            });
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

    /// The expression a call's callee is: its path, or the method call.
    fn callee_expr(&self, func: &Operand, call: MirSpan) -> Option<hir_def::hir::ExprId> {
        match (func.span(), call) {
            (Some(MirSpan::ExprId(expr)), _) => Some(expr),
            (_, MirSpan::ExprId(call)) => match self.ctx.store[call] {
                hir_def::hir::Expr::Call { callee, .. } => Some(callee),
                _ => Some(call),
            },
            _ => None,
        }
    }

    /// The type of a callee whose path spells out generic arguments, `f::<&'a T>`, with the
    /// lifetimes written there: rustc's user type annotation on the path.
    fn written_callee(&mut self, func: &Operand, call: MirSpan) -> Option<Ty<'db>> {
        let ty = match &func.kind {
            OperandKind::Constant { ty, .. } => ty.as_ref(),
            OperandKind::Allocation { allocation } => allocation.as_ref().ty,
            _ => return None,
        };
        let expr = self.callee_expr(func, call)?;
        match &self.ctx.store[expr] {
            hir_def::hir::Expr::Path(path) if spells_generics(path) => Some(self.written(ty)),
            // `x.f::<&'a T>()`: the method's own arguments as written, the impl's or trait's
            // inferred.
            hir_def::hir::Expr::MethodCall { generic_args: Some(_), .. } => {
                let TyKind::FnDef(def, args) = ty.kind() else { return None };
                let CallableDefId::FunctionId(function) = def.0 else { return None };
                let parent = hir_ty::generics::generics(self.ctx.db, function.into()).len_parent();
                let args = args.iter().enumerate().map(|(index, arg)| match index < parent {
                    true => self.renumber(arg),
                    false => self.written(arg),
                });
                let args = GenericArgs::new_from_iter(args.collect::<Vec<_>>());
                Some(Ty::new_fn_def(self.interner(), def, args))
            }
            _ => None,
        }
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

    /// rustc's `CopyBound`: a copied value's type is `Copy`, which `impl Copy for Foo<'static>`
    /// proves only for `'static`. References, pointers and scalars are `Copy` for any lifetime.
    fn copy_bound(&mut self, operand: &Operand, span: MirSpan) {
        let OperandKind::Copy(place) = &operand.kind else { return };
        let ty = self.place_ty(place.local, place.projection.as_slice()).ty;
        if !matches!(ty.kind(), TyKind::Adt(..) | TyKind::Tuple(..) | TyKind::Array(..)) {
            return;
        }
        let Some(copy) = self.interner().lang_items().Copy else { return };
        let trait_ref = rustc_type_ir::TraitRef::new(self.interner(), copy.into(), [ty]);
        let span = operand.span().unwrap_or(span);
        self.prove(trait_ref, At { category: Category::Predicate, span });
    }

    fn statement(&mut self, kind: &StatementKind, span: MirSpan) {
        kind.for_each_operand(&mut |operand| self.copy_bound(operand, span));
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
            Rvalue::Repeat(operand, len) => {
                if let (Some(ty), TyKind::Array(element, _)) =
                    (self.operand_ty(operand), dest.kind())
                {
                    self.relate(ty, element, Variance::Covariant, at);
                    // More than one copy of what is not a constant needs it `Copy`, which
                    // `impl Copy for Foo<'static>` proves only for `'static`.
                    let copies = hir_ty::consteval::try_const_usize(self.ctx.db, len.as_ref());
                    if !matches!(
                        operand.kind,
                        OperandKind::Constant { .. } | OperandKind::Allocation { .. }
                    ) && copies.is_none_or(|it| it > 1)
                        && let Some(copy) = self.interner().lang_items().Copy
                    {
                        let trait_ref =
                            rustc_type_ir::TraitRef::new(self.interner(), copy.into(), [ty]);
                        self.prove(trait_ref, At { category: Category::Predicate, span });
                    }
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
                let constant = match rvalue {
                    Rvalue::Use(operand) => self
                        .constant_ty(operand, span)
                        .or_else(|| self.written_fn_item(operand, span)),
                    _ => None,
                };
                if let Some(ty) = constant.or_else(|| self.rvalue_ty(rvalue, span)) {
                    self.relate(ty, dest, Variance::Covariant, at);
                }
            }
        }
    }

    /// The type of a named constant as rustc's type check gives it: its generic arguments
    /// renumbered, or kept as written where its path spells them out (rustc's user type
    /// annotation), with its where clauses proven; for a constant of an inherent impl named
    /// through a written self type, `<Foo<'a>>::C`, that type equal to the impl's.
    fn constant_ty(&mut self, operand: &Operand, span: MirSpan) -> Option<Ty<'db>> {
        let OperandKind::Constant { konst, .. } = &operand.kind else { return None };
        let ConstKind::Unevaluated(unevaluated) = konst.as_ref().kind() else { return None };
        let GeneralConstId::ConstId(id) = unevaluated.def.0 else { return None };
        let span = operand.span().unwrap_or(span);
        let path = match span {
            MirSpan::ExprId(expr) => match &self.ctx.store[expr] {
                hir_def::hir::Expr::Path(path) => Some(path),
                _ => None,
            },
            _ => None,
        };
        let args = match path {
            Some(path) if spells_generics(path) => self.written(unevaluated.args),
            _ => self.renumber(unevaluated.args),
        };
        let db = self.ctx.db;
        let ty = self
            .declared(db.value_ty(id.into())?.instantiate(self.interner(), args).skip_norm_wip());
        self.predicates(id.into(), args, At { category: Category::Predicate, span }, true);
        if let MirSpan::ExprId(expr) = span {
            self.path_well_formed(expr, id.lookup(db).container, args, span);
        }
        if let Some(anchor) = path.and_then(|it| it.type_anchor())
            && let ItemContainerId::ImplId(impl_) = id.lookup(db).container
        {
            let written = self.written(self.lower_written(anchor, span));
            let self_ty = self.declared(
                db.impl_self_ty(impl_).instantiate(self.interner(), args).skip_norm_wip(),
            );
            let at = At { category: Category::TypeAnnotation, span };
            self.relate(self_ty, written, Variance::Invariant, at);
        }
        Some(ty)
    }

    /// A function item used as a value whose path spells out generic arguments,
    /// `let f = <&'static u32 as Trait>::method;`, with the lifetimes written there and its
    /// where clauses proven, as rustc's user type annotation on the path.
    fn written_fn_item(&mut self, operand: &Operand, span: MirSpan) -> Option<Ty<'db>> {
        let ty = self.written_callee(operand, span)?;
        if let TyKind::FnDef(def, args) = ty.kind() {
            self.callee_predicates(def.0, args, operand.span().unwrap_or(span));
        }
        Some(ty)
    }

    /// A declared type instantiated with region variables: its `'static` the static region,
    /// any other region it names a fresh variable.
    fn declared(&self, ty: Ty<'db>) -> Ty<'db> {
        rustc_type_ir::fold_regions(self.interner(), ty, |region, _| match region.kind() {
            RegionKind::ReVar(_) | RegionKind::ReBound(..) => region,
            RegionKind::ReStatic => self.static_region(),
            _ => self.fresh(),
        })
    }

    /// A type written in the body, lowered as inference lowers it, its lifetimes the
    /// signature's where it names them.
    fn lower_written(&self, type_ref: hir_def::type_ref::TypeRefId, span: MirSpan) -> Ty<'db> {
        let db = self.ctx.db;
        let store_owner = self.ctx.owner.expression_store_owner(db);
        let mut resolver = store_owner.resolver(db);
        let _guard = match span {
            MirSpan::ExprId(expr) => Some(resolver.update_to_inner_scope(db, store_owner, expr)),
            _ => None,
        };
        let generics = std::cell::OnceCell::new();
        hir_ty::TyLoweringContext::new(
            db,
            &resolver,
            hir_def::expr_store::ExpressionStore::of(db, store_owner),
            store_owner,
            self.ctx.owner.generic_def(db),
            &generics,
            hir_ty::LifetimeElisionKind::Infer,
            hir_ty::LifetimeLoweringMode::LateParam,
        )
        .lower_ty(type_ref)
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
                let source = match kind {
                    CastKind::PointerCoercion(hir_ty::PointerCast::ReifyFnPointer) => {
                        self.written_callee(operand, span).or_else(|| self.operand_ty(operand))?
                    }
                    _ => self.operand_ty(operand)?,
                };
                let at = At { category: Category::Cast, span };
                match kind {
                    // rustc's casts between function types: what is cast is a subtype of the
                    // function pointer it becomes, by signature.
                    CastKind::PointerCoercion(hir_ty::PointerCast::UnsafeFnPointer) => {
                        self.relate(source, target, Variance::Covariant, at)
                    }
                    CastKind::PointerCoercion(hir_ty::PointerCast::ReifyFnPointer) => {
                        if let TyKind::FnDef(def, args) = source.kind() {
                            let sig = self
                                .ctx
                                .db
                                .callable_item_signature(def.0)
                                .instantiate(self.interner(), args)
                                .skip_norm_wip();
                            let source = Ty::new_fn_ptr(self.interner(), sig);
                            self.relate(source, target, Variance::Covariant, at);
                        }
                    }
                    CastKind::PointerCoercion(hir_ty::PointerCast::ClosureFnPointer(_)) => {
                        if let TyKind::Closure(_, args) = source.kind()
                            && let TyKind::FnPtr(to, _) = target.kind()
                            && let sig = args.as_closure().sig().skip_binder()
                            && let &[tupled] = sig.inputs().as_slice()
                            && let TyKind::Tuple(inputs) = tupled.kind()
                            && let Some((&output, to_inputs)) =
                                to.skip_binder().inputs_and_output.as_slice().split_last()
                        {
                            for (input, &to) in inputs.iter().zip(to_inputs) {
                                self.relate(input, to, Variance::Contravariant, at);
                            }
                            self.relate(sig.output(), output, Variance::Covariant, at);
                        }
                    }
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
                let store = self.ctx.store;
                let spelled = |expr| match &store[expr] {
                    hir_def::hir::Expr::RecordLit { path, .. } | hir_def::hir::Expr::Path(path) => {
                        spells_generics(path)
                    }
                    _ => false,
                };
                // A constructor called through a variable, `let f = Foo::<&'a u32>; f(x)`, has
                // the variable's arguments.
                let through_local = match span {
                    MirSpan::ExprId(expr) => match store[expr] {
                        hir_def::hir::Expr::Call { callee, .. } => self
                            .path_local(callee)
                            .map(|it| self.regions.local_tys[it.into_raw().into_u32() as usize])
                            .and_then(|it| match it.kind() {
                                TyKind::FnDef(def, args)
                                    if !matches!(def.0, CallableDefId::FunctionId(_)) =>
                                {
                                    Some(args)
                                }
                                _ => None,
                            }),
                        _ => None,
                    },
                    _ => None,
                };
                let args = match span {
                    _ if let Some(args) = through_local => args,
                    MirSpan::ExprId(expr)
                        if spelled(expr)
                            || matches!(store[expr],
                                hir_def::hir::Expr::Call { callee, .. } if spelled(callee)) =>
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
                self.predicates(adt.into(), args, boring, true);
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
                let written = ty.as_ref();
                let ty = self.renumber(written);
                let lifetimes = |parents: &[hir_ty::next_solver::GenericArg<'db>]| {
                    parents
                        .iter()
                        .enumerate()
                        .filter_map(|(index, arg)| match arg.kind() {
                            GenericArgKind::Lifetime(region) => Some((index as u32, region)),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                };
                let made = match ty.kind() {
                    TyKind::Closure(def, args) => Some((
                        def.0.loc(self.ctx.db).expr,
                        args.as_closure().tupled_upvars_ty(),
                        lifetimes(args.as_closure().parent_args().as_slice()),
                    )),
                    TyKind::Coroutine(def, args) => {
                        let expr = def.0.loc(self.ctx.db).expr;
                        if let TyKind::Coroutine(_, typed) = written.kind() {
                            self.async_fn_output(
                                expr,
                                typed.as_coroutine().return_ty(),
                                args.as_coroutine().return_ty(),
                                span,
                            );
                        }
                        Some((
                            expr,
                            args.as_coroutine().tupled_upvars_ty(),
                            lifetimes(args.as_coroutine().parent_args().as_slice()),
                        ))
                    }
                    _ => None,
                };
                if let Some((expr, upvars, parents)) = made {
                    if let TyKind::Tuple(upvars) = upvars.kind() {
                        for (upvar, operand) in upvars.iter().zip(operands.iter()) {
                            if let Some(it) = self.operand_ty(operand) {
                                let at = At {
                                    category: Category::Boring,
                                    span: operand.span().unwrap_or(span),
                                };
                                self.relate(it, upvar, Variance::Covariant, at);
                            }
                        }
                    }
                    self.closure_requirements(expr, ty, parents, span);
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
        kind.for_each_operand(&mut |operand| self.copy_bound(operand, span));
        match kind {
            TerminatorKind::Call { func, args, destination, .. } => {
                // A function item called through a variable, `let f = foo::<'a>; f()`, which
                // rust-analyzer lowers as a call of the constant, has the variable's type.
                let local = match (&func.kind, span) {
                    (
                        OperandKind::Constant { .. } | OperandKind::Allocation { .. },
                        MirSpan::ExprId(call),
                    ) => match self.ctx.store[call] {
                        hir_def::hir::Expr::Call { callee, .. } => self
                            .path_local(callee)
                            .filter(|&it| {
                                matches!(
                                    (self.regions.local_tys[it.into_raw().into_u32() as usize].kind(),
                                        self.operand_ty(func).map(|it| it.kind())),
                                    (TyKind::FnDef(a, _), Some(TyKind::FnDef(b, _))) if a == b
                                )
                            }),
                        _ => None,
                    },
                    _ => None,
                };
                let Some(func_ty) = local
                    .map(|it| self.regions.local_tys[it.into_raw().into_u32() as usize])
                    .or_else(|| self.written_callee(func, span))
                    .or_else(|| self.constant_ty(func, span))
                    .or_else(|| self.operand_ty(func))
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
                        if let CallableDefId::FunctionId(function) = def
                            && let Some(expr) = self.callee_expr(func, span)
                        {
                            let container = function.lookup(self.ctx.db).container;
                            self.path_well_formed(expr, container, fn_args, span);
                        }
                        let (self_param, keep_static, returns_self) = match def {
                            CallableDefId::FunctionId(it) => {
                                let written = FunctionSignature::of(self.ctx.db, it);
                                let returns_self = written.ret_type.is_some_and(|it| {
                                    matches!(&written.store[it],
                                        hir_def::hir::type_ref::TypeRef::Path(path)
                                            if path.is_self_type())
                                });
                                (
                                    written.has_self_param(),
                                    writes_static_object(&written.store),
                                    returns_self,
                                )
                            }
                            _ => (false, true, false),
                        };
                        let (inputs, output) = self.elide(
                            sig.inputs().iter().copied().collect(),
                            sig.output(),
                            self_param,
                            keep_static,
                            returns_self,
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
                // rustc builds a struct or a variant with a constructor's arguments as an
                // aggregate, whose fields are `Boring` to blame.
                let constructor = matches!(
                    func_ty.kind(),
                    TyKind::FnDef(def, _) if !matches!(def.0, CallableDefId::FunctionId(_))
                );
                let category = if constructor { Category::Boring } else { Category::CallArgument };
                for (arg, input) in args.iter().zip(inputs) {
                    if let Some(ty) = self.operand_ty(arg) {
                        let at = At { category, span: arg.span().unwrap_or(span) };
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
        returns_self: bool,
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
        // A return of `Self` elides nothing: its anonymous lifetimes are the impl's,
        // `impl Foo<'_>`, which rust-analyzer leaves as errors too.
        let output =
            rustc_type_ir::fold_regions(interner, output, |region, _| match region.kind() {
                RegionKind::ReError(_) if returns_self => self.fresh(),
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
        // Outliving every `'a` of the clause's own `for<'a>` is outliving `'static`.
        let forall = |region: Region<'db>| match region.kind() {
            RegionKind::ReBound(rustc_type_ir::BoundVarIndexKind::Bound(binder), _) => {
                binder == rustc_type_ir::INNERMOST
            }
            _ => false,
        };
        for clause in clauses {
            match clause.kind().skip_binder() {
                ClauseKind::RegionOutlives(OutlivesPredicate(longer, shorter)) => {
                    let shorter = match forall(shorter) {
                        true => self.static_region(),
                        false => shorter,
                    };
                    if let (Some(longer), Some(shorter)) =
                        (self.region_rv(longer), self.region_rv(shorter))
                    {
                        self.outlives(longer, shorter, at);
                    }
                }
                ClauseKind::TypeOutlives(OutlivesPredicate(ty, region))
                    if type_outlives
                        && !rustc_type_ir::TypeVisitableExt::has_escaping_bound_vars(&ty) =>
                {
                    let original = region;
                    let region = match region.kind() {
                        RegionKind::ReStatic => self.static_region(),
                        _ if forall(region) => self.static_region(),
                        _ => region,
                    };
                    let tests = self.regions.type_tests.len();
                    self.type_outlives(ty, region, at);
                    if forall(original) {
                        for test in &mut self.regions.type_tests[tests..] {
                            test.forall = true;
                        }
                    }
                }
                ClauseKind::Trait(predicate) if type_outlives => {
                    let trait_ref = predicate.trait_ref;
                    for region in self.trait_outlives(trait_ref) {
                        self.type_outlives(trait_ref.self_ty(), region, at);
                    }
                    self.caller_bound(trait_ref, at);
                    self.prove(trait_ref, at);
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

    /// Proves `trait_ref` with the trait solver, as rustc's `prove_predicate`, and takes the
    /// region constraints proving it gives: a where clause or an impl it matches relates the
    /// regions of the two (`&'min G: Get` proven by `&'max G: Get` equals them), and the
    /// impl's where clauses (`T: 'a`) hold. Nothing when it is not proven.
    fn prove(
        &mut self,
        trait_ref: rustc_type_ir::TraitRef<hir_ty::next_solver::DbInterner<'db>>,
        at: At,
    ) {
        use hir_ty::next_solver::{
            infer::{
                region_constraints::Constraint,
                traits::{Obligation, ObligationCause},
            },
            obligation_ctxt::ObligationCtxt,
        };
        if !rustc_type_ir::TypeVisitableExt::has_free_regions(&trait_ref)
            || rustc_type_ir::TypeVisitableExt::has_escaping_bound_vars(&trait_ref)
            || rustc_type_ir::TypeVisitableExt::references_error(&trait_ref)
            || trait_ref.args.iter().any(|arg| {
                rustc_type_ir::inherent::GenericArg::as_type(&arg).is_some_and(names_closure)
            })
        {
            return;
        }
        let interner = self.interner();
        let (constraints_before, obligations_before) = {
            let mut inner = self.ctx.infcx.inner.borrow_mut();
            let obligations = inner.region_obligations().len();
            let constraints =
                inner.unwrap_region_constraints().region_constraint_data().constraints.len();
            (constraints, obligations)
        };
        let mut ocx = ObligationCtxt::new(&self.ctx.infcx);
        let predicate = rustc_type_ir::TraitPredicate {
            trait_ref,
            polarity: rustc_type_ir::PredicatePolarity::Positive,
        };
        ocx.register_obligation(Obligation::new(
            interner,
            ObligationCause::dummy(),
            self.ctx.env,
            predicate,
        ));
        if !ocx.evaluate_obligations_error_on_ambiguity().is_empty() {
            return;
        }
        let (constraints, obligations) = {
            let mut inner = self.ctx.infcx.inner.borrow_mut();
            let obligations: Vec<_> = inner.region_obligations()[obligations_before..]
                .iter()
                .map(|it| (it.sup_type, it.sub_region))
                .collect();
            let constraints: Vec<_> =
                inner.unwrap_region_constraints().region_constraint_data().constraints
                    [constraints_before..]
                    .iter()
                    .cloned()
                    .collect();
            (constraints, obligations)
        };
        let rv = |this: &mut Self, region: Region<'db>| match region.kind() {
            RegionKind::ReVar(vid) => Some(vid.as_usize()),
            RegionKind::ReStatic => Some(this.regions.static_rv),
            RegionKind::ReEarlyParam(_) | RegionKind::ReLateParam(_) => {
                this.signature_region(region, rustc_type_ir::INNERMOST).and_then(rv_of)
            }
            _ => None,
        };
        for constraint in constraints {
            // `a <= b`: `b` outlives `a`.
            let (shorter, longer) = match constraint {
                Constraint::VarSubVar(a, b) => (Some(a.as_usize()), Some(b.as_usize())),
                Constraint::RegSubVar(a, b) => (rv(self, a), Some(b.as_usize())),
                Constraint::VarSubReg(a, b) => (Some(a.as_usize()), rv(self, b)),
                Constraint::RegSubReg(a, b) => (rv(self, a), rv(self, b)),
            };
            if let (Some(shorter), Some(longer)) = (shorter, longer) {
                self.outlives(longer, shorter, at);
            }
        }
        for (ty, region) in obligations {
            let Some(region) = rv(self, region) else { continue };
            let ty = rustc_type_ir::fold_regions(interner, ty, |it, _| match it.kind() {
                RegionKind::ReEarlyParam(_) | RegionKind::ReLateParam(_) | RegionKind::ReStatic => {
                    rv(self, it).map_or(it, |it| {
                        Region::new_var(interner, rustc_type_ir::RegionVid::from_usize(it))
                    })
                }
                _ => it,
            });
            self.type_outlives(
                ty,
                Region::new_var(interner, rustc_type_ir::RegionVid::from_usize(region)),
                at,
            );
        }
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
/// Whether `path` names the associated item `item` through a type, `Foo::f`, `<()>::f` or
/// `T::f`, not through its trait alone, `Default::default`: rustc gives it a user type
/// annotation whose generic arguments, and an inherent impl's self type, must be well-formed.
fn names_through_type(
    db: &dyn hir_ty::db::HirDatabase,
    path: &hir_def::expr_store::path::Path,
    container: ItemContainerId,
) -> bool {
    if path.type_anchor().is_some() {
        return true;
    }
    let segments = path.segments();
    let Some(qualifier) = segments.get(segments.len().wrapping_sub(2)) else { return false };
    match container {
        ItemContainerId::ImplId(_) => true,
        ItemContainerId::TraitId(trait_) => {
            *qualifier.name != hir_def::signatures::TraitSignature::of(db, trait_).name
        }
        _ => false,
    }
}

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
    // rustc's `best_blame_constraint` from a free region: the most interesting constraint,
    // and of equally interesting ones the closest to the region outlived.
    let interest = |it: &Outlives| match it.category {
        Category::Return => 0,
        Category::TypeAnnotation
        | Category::Cast
        | Category::CallArgument
        | Category::Assignment
        | Category::Usage => 2,
        Category::Predicate => 4,
        Category::Boring => 5,
        Category::Internal => 7,
    };
    let best = path.iter().map(interest).min()?;
    path.iter()
        .rev()
        .find(|it| interest(it) == best && it.span != MirSpan::Unknown)
        .or_else(|| path.iter().rev().find(|it| interest(it) == best))
        .copied()
}
