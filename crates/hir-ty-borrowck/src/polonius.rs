//! Loan liveness as rustc's `-Zpolonius=next` computes it: a loan is live where it can reach a
//! live region in a graph whose nodes are a region at a point. Edges within a point are the
//! outlives constraints that hold there; edges between points follow a region while it is
//! live, forwards for a covariant one, backwards for a contravariant one and both ways for an
//! invariant one. A loan is in scope until the first point it is not live at.

use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::{Variance, inherent::IntoKind};
use smallvec::SmallVec;

use hir_ty::{
    mir::MirBody,
    next_solver::{GenericArgKind, GenericArgs, Region, Ty, TyKind},
};

use super::{
    Ctx,
    bitset::BitSet,
    cfg::{Cfg, Location, index},
    loans::Loans,
    regions::{Regions, Rv, rv_of},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Direction {
    Forward,
    Backward,
    Both,
}

/// The points each loan is live at.
pub(super) fn live_loans<'db>(
    ctx: &Ctx<'_, 'db>,
    body: &MirBody<'db>,
    cfg: &Cfg,
    regions: &Regions<'db>,
    count: usize,
    liveness: &[BitSet],
    loans: &Loans,
) -> Vec<BitSet> {
    let mut directions: Vec<Option<Direction>> = vec![None; count];
    for &ty in &regions.local_tys {
        record_variances(ctx, ty, Variance::Covariant, &mut directions);
    }
    let universal: FxHashSet<Rv> = regions.universals.iter().map(|&(rv, _)| rv).collect();
    let direction = |rv: Rv| {
        if universal.contains(&rv) {
            Direction::Forward
        } else {
            // rustc falls back to both ways for a region it has no variance of.
            directions.get(rv).copied().flatten().unwrap_or(Direction::Both)
        }
    };
    let is_live = |rv: Rv, point: usize| {
        universal.contains(&rv) || liveness.get(rv).is_some_and(|it| it.contains(point))
    };
    let mut localized: FxHashMap<(Rv, usize), SmallVec<[Rv; 4]>> = FxHashMap::default();
    let mut logical: Vec<SmallVec<[Rv; 4]>> = vec![SmallVec::new(); count];
    for constraint in &regions.outlives {
        if constraint.sup >= count || constraint.sub >= count {
            continue;
        }
        let edges = match constraint.location {
            Some(location) => localized.entry((constraint.sup, cfg.point(location))).or_default(),
            None => &mut logical[constraint.sup],
        };
        if !edges.contains(&constraint.sub) {
            edges.push(constraint.sub);
        }
    }

    let mut live = Vec::with_capacity(loans.loans.len());
    let mut visited: FxHashSet<(Rv, usize)> = FxHashSet::default();
    let mut stack: Vec<(Rv, usize)> = Vec::new();
    for loan in &loans.loans {
        let mut points = BitSet::new(cfg.points);
        visited.clear();
        let start = (loan.region, cfg.point(loan.location));
        visited.insert(start);
        stack.push(start);
        while let Some((region, point)) = stack.pop() {
            let live_here = is_live(region, point);
            if live_here {
                points.insert(point);
            }
            let mut found = |node: (Rv, usize)| {
                if visited.insert(node) {
                    stack.push(node);
                }
            };
            for &sub in localized.get(&(region, point)).into_iter().flatten() {
                found((sub, point));
            }
            let location = cfg.location(point);
            let direction = direction(region);
            if matches!(direction, Direction::Forward | Direction::Both) {
                if location.statement < body.basic_blocks[location.block].statements.len() {
                    if is_live(region, point + 1) {
                        found((region, point + 1));
                    }
                } else {
                    for &successor in &cfg.successors[index(location.block)] {
                        let next = cfg.point(Location { block: successor, statement: 0 });
                        if is_live(region, next) {
                            found((region, next));
                        }
                    }
                }
            }
            if matches!(direction, Direction::Backward | Direction::Both) && live_here {
                for previous in cfg.predecessor_locations(body, location) {
                    found((region, cfg.point(previous)));
                }
            }
            for &sub in &logical[region] {
                found((sub, point));
            }
        }
        live.push(points);
    }
    live
}

/// Records the variance each region of `ty` has in it, as rustc's `VarianceExtractor`: a
/// region in two places of different variances is invariant.
fn record_variances<'db>(
    ctx: &Ctx<'_, 'db>,
    ty: Ty<'db>,
    variance: Variance,
    directions: &mut Vec<Option<Direction>>,
) {
    if variance == Variance::Bivariant {
        return;
    }
    match ty.kind() {
        TyKind::Ref(region, inner, mutability) => {
            record_region(region, variance, directions);
            let inner_variance = match mutability {
                rustc_ast_ir::Mutability::Not => variance,
                rustc_ast_ir::Mutability::Mut => variance.xform(Variance::Invariant),
            };
            record_variances(ctx, inner, inner_variance, directions);
        }
        TyKind::RawPtr(inner, mutability) => {
            let inner_variance = match mutability {
                rustc_ast_ir::Mutability::Not => variance,
                rustc_ast_ir::Mutability::Mut => variance.xform(Variance::Invariant),
            };
            record_variances(ctx, inner, inner_variance, directions);
        }
        TyKind::Adt(def, args) => {
            let variances = ctx.db.variances_of(hir_def::GenericDefId::AdtId(def.def_id()));
            record_args(ctx, args, Some(variances.as_slice()), variance, directions);
        }
        TyKind::Tuple(tys) => {
            for ty in tys.iter() {
                record_variances(ctx, ty, variance, directions);
            }
        }
        TyKind::Array(inner, _) | TyKind::Slice(inner) => {
            record_variances(ctx, inner, variance, directions)
        }
        TyKind::FnPtr(sig, _) => {
            let tys = sig.skip_binder().inputs_and_output;
            let last = tys.len().saturating_sub(1);
            for (index, ty) in tys.iter().enumerate() {
                let inner =
                    if index == last { variance } else { variance.xform(Variance::Contravariant) };
                record_variances(ctx, ty, inner, directions);
            }
        }
        TyKind::Dynamic(predicates, region) => {
            record_region(region, variance, directions);
            for predicate in predicates.iter() {
                if let rustc_type_ir::ExistentialPredicate::Trait(it) = predicate.skip_binder() {
                    record_args(
                        ctx,
                        it.args,
                        None,
                        variance.xform(Variance::Invariant),
                        directions,
                    );
                }
                if let rustc_type_ir::ExistentialPredicate::Projection(it) = predicate.skip_binder()
                {
                    record_args(
                        ctx,
                        it.args,
                        None,
                        variance.xform(Variance::Invariant),
                        directions,
                    );
                }
            }
        }
        TyKind::Alias(alias) => {
            record_args(ctx, alias.args, None, variance.xform(Variance::Invariant), directions)
        }
        TyKind::Closure(_, args)
        | TyKind::CoroutineClosure(_, args)
        | TyKind::Coroutine(_, args)
        | TyKind::CoroutineWitness(_, args)
        | TyKind::FnDef(_, args) => {
            record_args(ctx, args, None, variance.xform(Variance::Invariant), directions)
        }
        _ => (),
    }
}

fn record_args<'db>(
    ctx: &Ctx<'_, 'db>,
    args: GenericArgs<'db>,
    variances: Option<&[Variance]>,
    variance: Variance,
    directions: &mut Vec<Option<Direction>>,
) {
    for (index, arg) in args.iter().enumerate() {
        let param = variances.and_then(|it| it.get(index)).copied().unwrap_or(Variance::Invariant);
        let variance = variance.xform(param);
        match arg.kind() {
            GenericArgKind::Type(ty) => record_variances(ctx, ty, variance, directions),
            GenericArgKind::Lifetime(region) => record_region(region, variance, directions),
            GenericArgKind::Const(_) => (),
        }
    }
}

fn record_region(region: Region<'_>, variance: Variance, directions: &mut Vec<Option<Direction>>) {
    let Some(rv) = rv_of(region) else { return };
    let direction = match variance {
        Variance::Covariant => Direction::Forward,
        Variance::Contravariant => Direction::Backward,
        Variance::Invariant => Direction::Both,
        Variance::Bivariant => return,
    };
    if rv >= directions.len() {
        directions.resize(rv + 1, None);
    }
    let entry = &mut directions[rv];
    *entry = match *entry {
        Some(existing) if existing != direction => Some(Direction::Both),
        _ => Some(direction),
    };
}
