//! Liveness, as rustc's `type_check/liveness`: a region holds at each point where a local
//! whose type has it may still be used, and where a local may still be dropped while it may
//! be initialized, for the regions dropping it uses.

use hir_def::{AdtId, attrs::AttrFlags};
use rustc_hash::FxHashMap;
use rustc_type_ir::{
    ClauseKind, OutlivesPredicate, RegionKind,
    inherent::{
        AdtDef as _, GenericArgs as _, IntoKind, ParamEnv as _, Region as _, SliceLike as _,
    },
};
use smallvec::SmallVec;

use hir_ty::{
    mir::{
        BasicBlockId, LocalId, MirBody, Operand, OperandKind, PlaceElem, Rvalue, StatementKind,
        TerminatorKind,
    },
    next_solver::{Clause, DbInterner, GenericArgKind, Region, Ty, TyKind},
};

use super::{
    Ctx,
    bitset::BitSet,
    cfg::{Cfg, Location, index},
    moves::{InitKind, MoveData},
    regions::{Regions, Rv, regions_of, rv_of},
};

#[derive(Default)]
struct Effects {
    uses: Vec<LocalId>,
    defs: Vec<LocalId>,
    drops: Vec<LocalId>,
}

fn place_uses(local: LocalId, projection: &[PlaceElem], uses: &mut Vec<LocalId>) {
    uses.push(local);
    for elem in projection {
        if let PlaceElem::Index(index) = elem {
            uses.push(*index);
        }
    }
}

fn operand_uses(operand: &Operand, uses: &mut Vec<LocalId>) {
    if let OperandKind::Copy(place) | OperandKind::Move(place) = &operand.kind {
        place_uses(place.local, place.projection.as_slice(), uses);
    }
}

/// A write of a whole local defines it; a write of part of it uses it.
fn assigned(local: LocalId, projection: &[PlaceElem], effects: &mut Effects) {
    if projection.is_empty() {
        effects.defs.push(local);
    } else {
        place_uses(local, projection, &mut effects.uses);
    }
}

/// The values of expression statements (`match x { None => &temp(), Some(it) => it };`),
/// which lowering reads once the statement's temporaries are dropped, and rustc does not; with,
/// for a `match`, the type of its value, which tells that read from the read of what it matches
/// (they have the same span), where the two types differ.
fn statement_values<'db>(ctx: &Ctx<'_, 'db>) -> FxHashMap<hir_def::hir::ExprId, Option<Ty<'db>>> {
    let erased = |ty: Ty<'db>| {
        rustc_type_ir::fold_regions(ctx.infcx.interner, ty, |_, _| {
            Region::new_erased(ctx.infcx.interner)
        })
    };
    let mut found = FxHashMap::default();
    for (_, expr) in ctx.store.exprs() {
        let hir_def::hir::Expr::Block { statements, .. } = expr else { continue };
        for statement in statements.iter() {
            let hir_def::hir::Statement::Expr { expr, .. } = statement else { continue };
            match ctx.store[*expr] {
                hir_def::hir::Expr::Match { expr: matched, .. } => {
                    let value = erased(ctx.infer.expr_ty(*expr));
                    if value != erased(ctx.infer.expr_ty(matched)) {
                        found.insert(*expr, Some(value));
                    }
                }
                _ => {
                    found.insert(*expr, None);
                }
            }
        }
    }
    found
}

fn effects_at<'db>(
    ctx: &Ctx<'_, 'db>,
    body: &MirBody<'db>,
    location: Location,
    statement_values: &FxHashMap<hir_def::hir::ExprId, Option<Ty<'db>>>,
) -> Effects {
    let mut effects = Effects::default();
    let data = &body.basic_blocks[location.block];
    if let Some(statement) = data.statements.get(location.statement) {
        match &statement.kind {
            StatementKind::Assign(place, rvalue) => {
                match rvalue {
                    Rvalue::Ref(_, borrowed)
                    | Rvalue::Len(borrowed)
                    | Rvalue::Discriminant(borrowed)
                    | Rvalue::CopyForDeref(borrowed) => place_uses(
                        borrowed.local,
                        borrowed.projection.as_slice(),
                        &mut effects.uses,
                    ),
                    _ => rvalue.for_each_operand(&mut |it| operand_uses(it, &mut effects.uses)),
                }
                assigned(place.local, place.projection.as_slice(), &mut effects);
            }
            StatementKind::FakeRead(place)
                if let hir_ty::mir::MirSpan::ExprId(expr) = statement.span
                    && let Some(value) = statement_values.get(&expr)
                    && value.is_none_or(|value| {
                        place.projection.is_empty()
                            && rustc_type_ir::fold_regions(
                                ctx.infcx.interner,
                                body.locals[place.local].ty.as_ref(),
                                |_, _| Region::new_erased(ctx.infcx.interner),
                            ) == value
                    }) => {}
            StatementKind::FakeRead(place) => {
                place_uses(place.local, place.projection.as_slice(), &mut effects.uses)
            }
            StatementKind::Deinit(place) => {
                assigned(place.local, place.projection.as_slice(), &mut effects)
            }
            StatementKind::StorageLive(local) | StatementKind::StorageDead(local) => {
                effects.defs.push(*local)
            }
            StatementKind::InlineAsm(_) => {
                statement.kind.for_each_operand(&mut |it| operand_uses(it, &mut effects.uses))
            }
            StatementKind::Nop => (),
        }
    } else if let Some(terminator) = &data.terminator {
        match &terminator.kind {
            TerminatorKind::Drop { place, .. } => {
                if place.projection.is_empty() {
                    effects.drops.push(place.local);
                } else {
                    place_uses(place.local, place.projection.as_slice(), &mut effects.uses);
                }
            }
            TerminatorKind::DropAndReplace { place, value, .. } => {
                operand_uses(value, &mut effects.uses);
                place_uses(place.local, place.projection.as_slice(), &mut effects.uses);
            }
            TerminatorKind::Call { func, args, destination, .. } => {
                // The destination is written after the arguments are read.
                assigned(destination.local, destination.projection.as_slice(), &mut effects);
                operand_uses(func, &mut effects.uses);
                for arg in args.iter() {
                    operand_uses(arg, &mut effects.uses);
                }
            }
            TerminatorKind::Return => effects.uses.push(super::return_slot()),
            kind => kind.for_each_operand(&mut |it| operand_uses(it, &mut effects.uses)),
        }
    }
    effects
}

/// Which locals are live at the start of each point, by a backward analysis where `uses`
/// says which locals a point uses; `visit` sees each point once.
fn live_points(
    body: &MirBody<'_>,
    cfg: &Cfg,
    effects: &[Effects],
    uses: impl Fn(&Effects) -> &[LocalId],
    edge: impl Fn(BasicBlockId, BasicBlockId, &mut BitSet),
    mut visit: impl FnMut(usize, &BitSet),
) {
    let locals = body.locals.len();
    let blocks = body.basic_blocks.len();
    let mut entry = vec![BitSet::new(locals); blocks];
    let mut changed = true;
    while changed {
        changed = false;
        for &block in cfg.order.iter().rev() {
            let mut live = BitSet::new(locals);
            for &successor in &cfg.successors[index(block)] {
                let mut from = entry[index(successor)].clone();
                edge(block, successor, &mut from);
                live.union(&from);
            }
            let start = cfg.block_start(block);
            let end = cfg.block_end(body, block);
            for point in (start..=end).rev() {
                let it = &effects[point];
                for local in &it.defs {
                    live.remove(local.into_raw().into_u32() as usize);
                }
                for local in uses(it) {
                    live.insert(local.into_raw().into_u32() as usize);
                }
            }
            if entry[index(block)] != live {
                entry[index(block)] = live;
                changed = true;
            }
        }
    }
    for &block in &cfg.order {
        let mut live = BitSet::new(locals);
        for &successor in &cfg.successors[index(block)] {
            let mut from = entry[index(successor)].clone();
            edge(block, successor, &mut from);
            live.union(&from);
        }
        let start = cfg.block_start(block);
        let end = cfg.block_end(body, block);
        for point in (start..=end).rev() {
            let it = &effects[point];
            // A local is live where it is assigned if it is live after, as rustc's
            // `compute_use_live_points_for` has it.
            let mut assigned: SmallVec<[usize; 2]> = SmallVec::new();
            for local in &it.defs {
                let local = local.into_raw().into_u32() as usize;
                if live.contains(local) {
                    assigned.push(local);
                }
                live.remove(local);
            }
            for local in uses(it) {
                live.insert(local.into_raw().into_u32() as usize);
            }
            assigned.retain(|it| live.insert(*it));
            visit(point, &live);
            for local in assigned {
                live.remove(local);
            }
        }
    }
}

/// The value of each region from liveness alone.
pub(super) fn region_liveness<'db>(
    ctx: &Ctx<'_, 'db>,
    body: &MirBody<'db>,
    cfg: &Cfg,
    moves: &MoveData,
    regions: &Regions<'db>,
    count: usize,
) -> Vec<BitSet> {
    let size = cfg.points + regions.universals.len();
    let mut values = vec![BitSet::new(size); count];
    let mut effects: Vec<Effects> = (0..cfg.points).map(|_| Effects::default()).collect();
    let statement_values = statement_values(ctx);
    for &block in &cfg.order {
        for point in cfg.block_start(block)..=cfg.block_end(body, block) {
            effects[point] = effects_at(ctx, body, cfg.location(point), &statement_values);
        }
    }
    for &(local, location) in &regions.callee_reads {
        effects[cfg.point(location)].uses.push(local);
    }
    // A drop of what cannot be initialized there does nothing.
    let (maybe_init, init_at_exit) = maybe_initialized_at_drops(body, cfg, moves);
    for (point, it) in effects.iter_mut().enumerate() {
        it.drops.retain(|&local| maybe_init.get(&point).is_some_and(|set| set.contains(&local)));
    }
    let local_regions: Vec<Vec<Rv>> = regions
        .local_tys
        .iter()
        .map(|&ty| {
            let mut out = Vec::new();
            live_regions_of(ctx, regions, ty, &mut out);
            out
        })
        .collect();
    let mut use_live: Vec<BitSet> = vec![BitSet::new(0); cfg.points];
    live_points(
        body,
        cfg,
        &effects,
        |it| &it.uses,
        |_, _, _| (),
        |point, live| {
            use_live[point] = live.clone();
            for local in live.iter() {
                for &rv in &local_regions[local] {
                    if rv < count {
                        values[rv].insert(point);
                    }
                }
            }
        },
    );
    let mut drop_regions: FxHashMap<usize, Vec<Rv>> = FxHashMap::default();
    let mut cache = FxHashMap::default();
    live_points(
        body,
        cfg,
        &effects,
        |it| &it.drops,
        // Dropping a local makes it live back to where it may be initialized only, as
        // rustc's `compute_drop_live_points_for` has it.
        |block, successor, live| live.intersect(init_at_exit.at(block, successor)),
        |point, live| {
            for local in live.iter() {
                if use_live[point].contains(local) {
                    continue;
                }
                let rvs = drop_regions.entry(local).or_insert_with(|| {
                    let mut out = Vec::new();
                    drop_components(
                        ctx,
                        regions,
                        regions.local_tys[local],
                        &mut out,
                        &mut cache,
                        0,
                    );
                    out
                });
                for &rv in rvs.iter() {
                    if rv < count {
                        values[rv].insert(point);
                    }
                }
            }
        },
    );
    values
}

/// For each drop, the locals it drops that may be initialized there; and for each edge
/// between blocks, the locals that may be initialized, if partly, leaving the first for the
/// second.
fn maybe_initialized_at_drops(
    body: &MirBody<'_>,
    cfg: &Cfg,
    moves: &MoveData,
) -> (FxHashMap<usize, Vec<LocalId>>, InitAtExit) {
    let paths = moves.paths.len();
    let blocks = body.basic_blocks.len();
    let mut entry: Vec<Option<BitSet>> = vec![None; blocks];
    let mut start = BitSet::new(paths);
    for init in moves.inits.iter().filter(|it| it.location.is_none()) {
        moves.with_descendants(init.path, &mut |it| {
            start.insert(it);
        });
    }
    entry[index(body.start_block)] = Some(start);
    let apply = |location: Location, state: &mut BitSet, returned: bool| {
        if !returned {
            if let Some(at) = moves.moves_at.get(&location) {
                for &it in at {
                    moves.with_descendants(moves.moves[it].path, &mut |path| state.remove(path));
                }
            }
        }
        if let Some(at) = moves.inits_at.get(&location) {
            for &it in at {
                let init = moves.inits[it];
                if (init.kind == InitKind::Deep) != returned {
                    moves.with_descendants(init.path, &mut |path| {
                        state.insert(path);
                    });
                }
            }
        }
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &block in &cfg.order {
            let Some(mut state) = entry[index(block)].clone() else { continue };
            let data = &body.basic_blocks[block];
            for statement in 0..=data.statements.len() {
                apply(Location { block, statement }, &mut state, false);
            }
            let returned = match data.terminator.as_ref().map(|it| &it.kind) {
                Some(TerminatorKind::Call { target: Some(target), .. }) => Some(*target),
                _ => None,
            };
            for &successor in &cfg.successors[index(block)] {
                let mut state = state.clone();
                if Some(successor) == returned {
                    apply(Location { block, statement: data.statements.len() }, &mut state, true);
                }
                match &mut entry[index(successor)] {
                    Some(it) => changed |= it.union(&state),
                    slot @ None => {
                        *slot = Some(state);
                        changed = true;
                    }
                }
            }
        }
    }
    let locals = body.locals.len();
    // Every path of a local is below the local's own, so a local is initialized in part where
    // one of its paths is.
    let initialized = |state: &BitSet| {
        let mut out = BitSet::new(locals);
        for path in state.iter() {
            out.insert(moves.paths[path].local.into_raw().into_u32() as usize);
        }
        out
    };
    let empty = BitSet::new(locals);
    let mut exits: Vec<Option<(BitSet, Option<(BasicBlockId, BitSet)>)>> = vec![None; blocks];
    let mut found: FxHashMap<usize, Vec<LocalId>> = FxHashMap::default();
    for &block in &cfg.order {
        let Some(mut state) = entry[index(block)].clone() else { continue };
        let data = &body.basic_blocks[block];
        for statement in 0..data.statements.len() {
            apply(Location { block, statement }, &mut state, false);
        }
        let mut exit = state.clone();
        apply(Location { block, statement: data.statements.len() }, &mut exit, false);
        let returned = match data.terminator.as_ref().map(|it| &it.kind) {
            Some(TerminatorKind::Call { target: Some(target), .. }) => {
                let mut returned = exit.clone();
                apply(Location { block, statement: data.statements.len() }, &mut returned, true);
                Some((*target, initialized(&returned)))
            }
            _ => None,
        };
        exits[index(block)] = Some((initialized(&exit), returned));
        if let Some(TerminatorKind::Drop { place, .. }) =
            data.terminator.as_ref().map(|it| &it.kind)
            && place.projection.is_empty()
        {
            let path = moves.local_path(place.local);
            if moves.find_in_descendants(path, &|it| state.contains(it)).is_some() {
                let point = cfg.point(Location { block, statement: data.statements.len() });
                found.entry(point).or_default().push(place.local);
            }
        }
    }
    (found, InitAtExit { exits, empty })
}

/// The locals that may be initialized, if partly, leaving a block for a successor.
struct InitAtExit {
    exits: Vec<Option<(BitSet, Option<(BasicBlockId, BitSet)>)>>,
    empty: BitSet,
}

impl InitAtExit {
    fn at(&self, block: BasicBlockId, successor: BasicBlockId) -> &BitSet {
        match &self.exits[index(block)] {
            Some((_, Some((target, returned)))) if *target == successor => returned,
            Some((exit, _)) => exit,
            None => &self.empty,
        }
    }
}

/// The regions dropping a value of type `ty` may use: rustc's dropck outlives constraints.
fn drop_components<'db>(
    ctx: &Ctx<'_, 'db>,
    regions: &Regions<'db>,
    ty: Ty<'db>,
    out: &mut Vec<Rv>,
    cache: &mut FxHashMap<Ty<'db>, Vec<Rv>>,
    depth: usize,
) {
    if depth > 12 {
        return;
    }
    if let Some(it) = cache.get(&ty) {
        out.extend_from_slice(it);
        return;
    }
    let mut found = Vec::new();
    match ty.kind() {
        // Dropping a `ManuallyDrop` drops nothing.
        TyKind::Adt(def, _) if def.is_manually_drop() => (),
        TyKind::Adt(def, args) => {
            let adt = def.def_id();
            if def.is_box() || def.is_phantom_data() {
                if let Some(GenericArgKind::Type(inner)) = args.iter().next().map(|it| it.kind()) {
                    drop_components(ctx, regions, inner, &mut found, cache, depth + 1);
                }
            } else {
                if let Some(drop_impl) = hir_ty::drop::destructor(ctx.db, adt) {
                    let dangling = may_dangle(ctx, *drop_impl, args.len());
                    for (index, arg) in args.iter().enumerate() {
                        let dangles = dangling.get(index).copied().unwrap_or(false);
                        match arg.kind() {
                            GenericArgKind::Type(arg) if dangles => {
                                drop_components(ctx, regions, arg, &mut found, cache, depth + 1)
                            }
                            GenericArgKind::Type(arg) => regions_of(arg, &mut found),
                            GenericArgKind::Lifetime(region) if !dangles => {
                                found.extend(super::regions::rv_of(region))
                            }
                            _ => (),
                        }
                    }
                }
                let variants: Vec<hir_def::VariantId> = match adt {
                    AdtId::StructId(it) => vec![it.into()],
                    AdtId::UnionId(_) => Vec::new(),
                    AdtId::EnumId(it) => it
                        .enum_variants(ctx.db)
                        .variants
                        .values()
                        .map(|&(it, _)| it.into())
                        .collect(),
                };
                for variant in variants {
                    for (_, field) in ctx.db.field_types(variant).iter() {
                        let field =
                            field.ty().instantiate(ctx.infcx.interner, args).skip_norm_wip();
                        drop_components(ctx, regions, field, &mut found, cache, depth + 1);
                    }
                }
            }
        }
        TyKind::Array(inner, _) | TyKind::Slice(inner) => {
            drop_components(ctx, regions, inner, &mut found, cache, depth + 1)
        }
        TyKind::Tuple(tys) => {
            for it in tys.iter() {
                drop_components(ctx, regions, it, &mut found, cache, depth + 1);
            }
        }
        TyKind::Closure(_, args) => drop_components(
            ctx,
            regions,
            args.as_closure().tupled_upvars_ty(),
            &mut found,
            cache,
            depth + 1,
        ),
        // rustc's dropck drops what a coroutine captures too, all of it where what it keeps
        // across an `await` needs dropping, which is not known here.
        TyKind::CoroutineClosure(_, args) => drop_components(
            ctx,
            regions,
            args.as_coroutine_closure().tupled_upvars_ty(),
            &mut found,
            cache,
            depth + 1,
        ),
        TyKind::Coroutine(_, args) => drop_components(
            ctx,
            regions,
            args.as_coroutine().tupled_upvars_ty(),
            &mut found,
            cache,
            depth + 1,
        ),
        TyKind::Dynamic(..) => regions_of(ty, &mut found),
        // A projection is what it normalizes to, `NoisyDrop<T>` of `<T as Associator>::As`;
        // another alias holds what its bounds and captures let it, as for liveness.
        TyKind::Alias(alias) => {
            let normalized = match alias.kind {
                rustc_type_ir::AliasTyKind::Projection { .. } => super::regions::normalize(ctx, ty)
                    .filter(|it| !matches!(it.kind(), TyKind::Alias(_) | TyKind::Infer(_))),
                _ => None,
            };
            match normalized {
                Some(it) => drop_components(ctx, regions, it, &mut found, cache, depth + 1),
                None => live_regions_of(ctx, regions, ty, &mut found),
            }
        }
        _ => (),
    }
    found.sort_unstable();
    found.dedup();
    out.extend_from_slice(&found);
    cache.insert(ty, found);
}

/// Per generic argument of the type a `Drop` impl is for, whether the impl says
/// `#[may_dangle]` of it: that dropping does not use it.
fn may_dangle(ctx: &Ctx<'_, '_>, drop_impl: hir_def::ImplId, count: usize) -> Vec<bool> {
    let (lifetimes, types) = AttrFlags::query_generic_params(ctx.db, drop_impl.into());
    let self_ty = ctx.db.impl_self_ty(drop_impl).skip_binder();
    let TyKind::Adt(_, args) = self_ty.kind() else { return vec![false; count] };
    args.iter()
        .map(|arg| match arg.kind() {
            GenericArgKind::Type(ty) => match ty.kind() {
                TyKind::Param(param) => types
                    .get(param.id.local_id())
                    .is_some_and(|it| it.contains(AttrFlags::MAY_DANGLE)),
                _ => false,
            },
            GenericArgKind::Lifetime(region) => match region.kind() {
                rustc_type_ir::RegionKind::ReEarlyParam(early) => lifetimes
                    .get(early.id.local_id)
                    .is_some_and(|it| it.contains(AttrFlags::MAY_DANGLE)),
                _ => false,
            },
            GenericArgKind::Const(_) => false,
        })
        .collect()
}

/// The regions of `ty` a value of it can hold on to, as rustc's liveness `FreeRegionsVisitor`
/// has them: an alias known to outlive a region holds only that one, and none when it
/// outlives `'static`.
fn live_regions_of<'db>(
    ctx: &Ctx<'_, 'db>,
    regions: &Regions<'db>,
    ty: Ty<'db>,
    out: &mut Vec<Rv>,
) {
    struct Visit<'c, 'a, 'db> {
        ctx: &'c Ctx<'a, 'db>,
        regions: &'c Regions<'db>,
        out: &'c mut Vec<Rv>,
    }
    impl<'db> rustc_type_ir::TypeVisitor<DbInterner<'db>> for Visit<'_, '_, 'db> {
        type Result = ();
        fn visit_region(&mut self, region: Region<'db>) {
            if let Some(rv) = rv_of(region) {
                self.out.push(rv);
            }
        }
        fn visit_ty(&mut self, ty: Ty<'db>) {
            if let TyKind::Alias(alias) = ty.kind() {
                match alias_bound(self.ctx, self.regions, ty, alias) {
                    Some(None) => return,
                    Some(Some(rv)) => {
                        self.out.push(rv);
                        return;
                    }
                    None => (),
                }
                // What an `impl Trait` does not capture need not be live.
                if let rustc_type_ir::AliasTyKind::Opaque { .. } = alias.kind
                    && let Some(captured) = super::regions::opaque_captures(
                        self.ctx.db,
                        alias.kind.def_id(),
                        alias.args.len(),
                    )
                {
                    for (arg, captured) in alias.args.iter().zip(captured) {
                        if captured {
                            rustc_type_ir::TypeVisitable::visit_with(&arg, self);
                        }
                    }
                    return;
                }
            }
            rustc_type_ir::TypeSuperVisitable::super_visit_with(&ty, self)
        }
    }
    rustc_type_ir::TypeVisitable::visit_with(&ty, &mut Visit { ctx, regions, out });
}

/// The one region the bounds of the alias `ty` say it outlives: `Some(None)` for `'static`,
/// `None` when they name none or several, or when `ty` is under a binder of its own type,
/// `fn(T::Gat<'a, '_>)`, whose bounds rustc does not match.
fn alias_bound<'db>(
    ctx: &Ctx<'_, 'db>,
    regions: &Regions<'db>,
    ty: Ty<'db>,
    alias: rustc_type_ir::AliasTy<DbInterner<'db>>,
) -> Option<Option<Rv>> {
    if rustc_type_ir::TypeVisitableExt::has_escaping_bound_vars(&ty) {
        return None;
    }
    let interner = ctx.infcx.interner;
    let erase = |ty: Ty<'db>| {
        rustc_type_ir::fold_regions(interner, ty, |_, _| Region::new_static(interner))
    };
    let erased = erase(ty);
    let item_bounds: Vec<Clause<'db>> =
        rustc_type_ir::Interner::item_bounds(interner, alias.kind.def_id())
            .iter_instantiated(interner, alias.args)
            .map(|it| it.skip_norm_wip())
            .collect();
    let mut found: Option<Option<Rv>> = None;
    for clause in item_bounds.into_iter().chain(ctx.env.caller_bounds().iter()) {
        let ClauseKind::TypeOutlives(OutlivesPredicate(bounded, region)) =
            clause.kind().skip_binder()
        else {
            continue;
        };
        let same = erase(bounded) == erased;
        if !same && normalize_erased(ctx, erase(bounded)) != Some(erased) {
            continue;
        }
        let bound = match region.kind() {
            RegionKind::ReStatic => return Some(None),
            // `for<'a> T: 'a` is `T: 'static`; `for<'a, 'b> T::Changes<'a, 'b>: 'b` that the
            // alias outlives what it has where the bound names `'b`.
            RegionKind::ReBound(_, bound) => {
                let regions_in = |ty: Ty<'db>| {
                    let mut found = Vec::new();
                    rustc_type_ir::fold_regions(interner, ty, |region, _| {
                        found.push(region);
                        region
                    });
                    found
                };
                let at = regions_in(bounded).iter().position(|it| {
                    matches!(it.kind(), RegionKind::ReBound(_, other) if other.var == bound.var)
                });
                match at {
                    None => return Some(None),
                    Some(at) if same => regions_in(ty).get(at).copied().and_then(rv_of),
                    Some(_) => None,
                }
            }
            RegionKind::ReVar(_) => rv_of(region),
            RegionKind::ReEarlyParam(early) => regions.early.get(&early.index).copied(),
            _ => None,
        };
        let Some(bound) = bound else { continue };
        match found {
            None => found = Some(Some(bound)),
            Some(Some(it)) if it == bound => (),
            Some(_) => return None,
        }
    }
    found
}

/// `ty`, its regions all `'static`, normalized as far as the solver goes: rustc normalizes the
/// outlives clauses of the parameter environment before alias liveness matches them.
pub(super) fn normalize_erased<'db>(ctx: &Ctx<'_, 'db>, ty: Ty<'db>) -> Option<Ty<'db>> {
    use hir_ty::next_solver::{infer::traits::ObligationCause, obligation_ctxt::ObligationCtxt};
    if !rustc_type_ir::TypeVisitableExt::has_aliases(&ty) {
        return None;
    }
    let infcx = &ctx.infcx;
    infcx.probe(|_| {
        let ocx = ObligationCtxt::new(infcx);
        let normalized = ocx.deeply_normalize(&ObligationCause::dummy(), ctx.env, ty).ok()?;
        let interner = infcx.interner;
        Some(rustc_type_ir::fold_regions(
            interner,
            infcx.resolve_vars_if_possible(normalized),
            |_, _| Region::new_static(interner),
        ))
    })
}
