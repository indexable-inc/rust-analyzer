//! Loans, as rustc's `borrow_set.rs`, the `Borrows` analysis of `dataflow.rs` and
//! `places_conflict.rs`: each borrow is a loan of a place, in scope from where it is made to
//! where its region stops holding, or the reference to it is overwritten, or the place it
//! borrows goes away.

use hir_def::{
    DefWithBodyId,
    hir::{Array, Expr},
};
use rustc_hash::{FxHashMap, FxHashSet};
use rustc_type_ir::inherent::IntoKind;

use hir_ty::{
    InferBodyId,
    mir::{
        BorrowKind, LocalId, MirBody, MirSpan, MutBorrowKind, OperandKind, PlaceElem,
        ProjectionElem, Rvalue, StatementKind, TerminatorKind,
    },
    next_solver::TyKind,
};

use super::{
    Ctx,
    bitset::BitSet,
    cfg::{Cfg, Location, index},
    regions::{RegionValues, Regions, Rv},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Activation {
    NotTwoPhase,
    NotActivated,
    At(Location),
}

pub(super) struct Loan {
    pub(super) local: LocalId,
    pub(super) projection: Vec<PlaceElem>,
    pub(super) kind: BorrowKind,
    pub(super) region: Rv,
    pub(super) location: Location,
    pub(super) span: MirSpan,
    /// The local the reference is assigned to.
    pub(super) assigned: LocalId,
    pub(super) activation: Activation,
}

pub(super) struct Loans {
    pub(super) loans: Vec<Loan>,
    pub(super) at: FxHashMap<Location, usize>,
    pub(super) of_local: FxHashMap<LocalId, Vec<usize>>,
    pub(super) activations: FxHashMap<Location, Vec<usize>>,
    /// The loans that go out of scope at the start of each point.
    out_of_scope: FxHashMap<usize, Vec<usize>>,
}

/// How much of a place an access touches, as rustc's `AccessDepth`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Depth {
    /// The place, not what is behind references in it, as an assignment.
    Shallow,
    /// The length of an array, or what a fake borrow sees.
    ShallowArtificial,
    Deep,
    /// A drop, which does not reach behind a reference.
    Drop,
}

/// How to take two places it is not known whether they overlap, as two indexes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Bias {
    Overlap,
    NoOverlap,
}

/// Whether a borrow of `place` is not a loan rustc tracks: one through a shared reference or
/// a raw pointer, which nothing the body does can invalidate.
fn ignored<'db>(
    ctx: &Ctx<'_, 'db>,
    body: &MirBody<'db>,
    local: LocalId,
    projection: &[PlaceElem],
    statics: &FxHashSet<LocalId>,
) -> bool {
    if statics.contains(&local) {
        return true;
    }
    for (index, elem) in projection.iter().enumerate() {
        if *elem == ProjectionElem::Deref {
            let ty = ctx.place_ty(body, local, &projection[..index]).ty;
            match ty.kind() {
                TyKind::Ref(_, _, rustc_ast_ir::Mutability::Not) | TyKind::RawPtr(..) => {
                    return true;
                }
                _ => (),
            }
        }
    }
    false
}

/// Whether `expr` calls a function, which rustc does not promote, not even a `const fn` in a
/// `const` block; a tuple struct's or variant's constructor is not one, nor is a
/// `#[rustc_promotable]` function such as `size_of`.
fn is_call(ctx: &Ctx<'_, '_>, expr: hir_def::hir::ExprId) -> bool {
    match ctx.store[expr] {
        Expr::MethodCall { .. } => true,
        Expr::Call { callee, .. } => !matches!(
            ctx.infer.expr_ty(callee).kind(),
            TyKind::FnDef(def, _) if match def.0 {
                hir_def::CallableDefId::StructId(_) | hir_def::CallableDefId::EnumVariantId(_) => true,
                hir_def::CallableDefId::FunctionId(func) => hir_def::attrs::AttrFlags::query(ctx.db, func.into())
                    .contains(hir_def::attrs::AttrFlags::RUSTC_PROMOTABLE),
            }
        ),
        _ => false,
    }
}

/// Whether `expr` is a unit struct or variant of a type with no `Drop` impl: what rustc
/// promotes whatever the rest of the type holds, as it asks that of the value.
fn is_unit_value(ctx: &Ctx<'_, '_>, expr: hir_def::hir::ExprId) -> bool {
    matches!(ctx.store[expr], Expr::Path(_))
        && ctx.infer.variant_resolution_for_expr(expr).is_some()
        && matches!(ctx.infer.expr_ty(expr).kind(),
            TyKind::Adt(def, _) if hir_ty::drop::destructor(ctx.db, def.def_id()).is_none())
}

/// Whether `expr` names a `const` item outside any impl or trait: rustc asks whether its value,
/// not its type, needs dropping, which only a value that does (a `String` that is not empty)
/// fails; such a value is taken to be promotable when its type has no interior mutability.
fn names_const_item(ctx: &Ctx<'_, '_>, body: &MirBody<'_>, expr: hir_def::hir::ExprId) -> bool {
    use hir_def::{
        Lookup,
        resolver::{HasResolver, ValueNs},
    };
    let Expr::Path(path) = &ctx.store[expr] else { return false };
    let owner = body.owner;
    let mut resolver = owner.resolver(ctx.db);
    let _guard = resolver.update_to_inner_scope(ctx.db, owner.expression_store_owner(ctx.db), expr);
    let hygiene = ctx.store.expr_path_hygiene(expr);
    matches!(
        resolver.resolve_path_in_value_ns_fully(ctx.db, path, hygiene),
        Some(ValueNs::ConstId(it))
            if matches!(it.lookup(ctx.db).container, hir_def::ItemContainerId::ModuleId(_))
    )
}

/// The locals a `static` is read through.
pub(super) fn static_locals(body: &MirBody<'_>) -> FxHashSet<LocalId> {
    let mut found = FxHashSet::default();
    for (_, block) in body.basic_blocks.iter() {
        for statement in &block.statements {
            if let StatementKind::Assign(place, Rvalue::Use(operand)) = &statement.kind
                && let OperandKind::Static(_) = operand.kind
                && place.projection.is_empty()
            {
                found.insert(place.local);
            }
        }
    }
    found
}

/// The temporaries a `const` or `static` item's value borrows where rustc extends their
/// lifetime, into the value itself: the operand of a `&` the value is, or that is an element,
/// a field, a block's value or a cast operand of such.
fn extended_temporaries(ctx: &Ctx<'_, '_>, body: &MirBody<'_>) -> FxHashSet<LocalId> {
    let mut refs = FxHashSet::default();
    let mut stack: Vec<_> = ctx.store.expr_roots().next_back().into_iter().collect();
    while let Some(expr) = stack.pop() {
        match &ctx.store[expr] {
            Expr::Ref { expr: inner, .. } => {
                refs.insert(expr);
                stack.push(*inner);
            }
            Expr::Array(Array::ElementList { elements }) => stack.extend(elements.iter().copied()),
            Expr::Tuple { exprs } => stack.extend(exprs.iter().copied()),
            Expr::RecordLit { fields, .. } => stack.extend(fields.iter().map(|it| it.expr)),
            Expr::Block { tail: Some(tail), .. } => stack.push(*tail),
            Expr::Cast { expr, .. } => stack.push(*expr),
            _ => (),
        }
    }
    let mut found = FxHashSet::default();
    for (_, block) in body.basic_blocks.iter() {
        for statement in &block.statements {
            if let StatementKind::Assign(_, Rvalue::Ref(_, place)) = &statement.kind
                && let MirSpan::ExprId(expr) = statement.span
                && refs.contains(&expr)
                && place.projection.is_empty()
            {
                found.insert(place.local);
            }
        }
    }
    found
}

impl Loans {
    pub(super) fn gather<'db>(
        ctx: &Ctx<'_, 'db>,
        body: &MirBody<'db>,
        cfg: &Cfg,
        regions: &Regions<'_>,
        unique: impl Fn(LocalId, LocalId, &[PlaceElem]) -> bool,
    ) -> Self {
        let mut statics = static_locals(body);
        // A promoted constant is `'static`, as a `static` is; so is what a `const` or `static`
        // item's value borrows, which becomes part of that value whatever it calls.
        let item = matches!(
            body.owner,
            InferBodyId::DefWithBodyId(DefWithBodyId::ConstId(_) | DefWithBodyId::StaticId(_))
        );
        statics.extend(
            body.promoted_locals
                .iter()
                .filter(|&&(local, expr)| {
                    // rustc promotes no call, not even of a `const fn` in a `const` block.
                    item || (!is_call(ctx, expr)
                        && (ctx.is_plain(body.locals[local].ty.as_ref())
                            || is_unit_value(ctx, expr)
                            || (names_const_item(ctx, body, expr)
                                && hir_ty::drop::is_freeze(
                                    ctx.db,
                                    ctx.infcx.interner,
                                    body.locals[local].ty.as_ref(),
                                ))))
                })
                .map(|&(local, _)| local),
        );
        if item {
            statics.extend(extended_temporaries(ctx, body));
        }
        // A function item is a constant of a type with no data, promoted where it is borrowed:
        // `&Trait::method`, which inference may leave without a type when it cannot tell `Self`.
        for (_, block) in body.basic_blocks.iter() {
            for statement in &block.statements {
                if let StatementKind::Assign(place, Rvalue::Use(operand)) = &statement.kind
                    && !matches!(operand.kind, OperandKind::Copy(_) | OperandKind::Move(_))
                    && place.projection.is_empty()
                    && matches!(
                        body.locals[place.local].ty.as_ref().kind(),
                        TyKind::FnDef(..) | TyKind::Error(_)
                    )
                {
                    statics.insert(place.local);
                }
            }
        }
        let mut loans = Loans {
            loans: Vec::new(),
            at: FxHashMap::default(),
            of_local: FxHashMap::default(),
            activations: FxHashMap::default(),
            out_of_scope: FxHashMap::default(),
        };
        let mut two_phase: FxHashMap<LocalId, usize> = FxHashMap::default();
        for &block in &cfg.order {
            for (statement, it) in body.basic_blocks[block].statements.iter().enumerate() {
                let StatementKind::Assign(place, Rvalue::Ref(kind, borrowed)) = &it.kind else {
                    continue;
                };
                let location = Location { block, statement };
                let projection = borrowed.projection.as_slice();
                if ignored(ctx, body, borrowed.local, projection, &statics) {
                    continue;
                }
                let Some(&region) = regions.loan_regions.get(&location) else { continue };
                let mut kind = *kind;
                // rustc makes the mutable borrow a closure captures of a variable without
                // `mut` a unique one.
                if let BorrowKind::Mut { kind: MutBorrowKind::Default } = kind
                    && unique(place.local, borrowed.local, projection)
                {
                    kind = BorrowKind::Mut { kind: MutBorrowKind::ClosureCapture };
                }
                let two = matches!(kind, BorrowKind::Mut { kind: MutBorrowKind::TwoPhasedBorrow })
                    && place.projection.is_empty();
                let id = loans.loans.len();
                if two {
                    two_phase.insert(place.local, id);
                }
                loans.loans.push(Loan {
                    local: borrowed.local,
                    projection: projection.to_vec(),
                    kind,
                    region,
                    location,
                    span: it.span,
                    assigned: place.local,
                    activation: if two {
                        Activation::NotActivated
                    } else {
                        Activation::NotTwoPhase
                    },
                });
                loans.at.insert(location, id);
                loans.of_local.entry(borrowed.local).or_default().push(id);
            }
        }
        if !two_phase.is_empty() {
            // A two-phase borrow is activated by the first use of the reference.
            for &block in &cfg.order {
                let data = &body.basic_blocks[block];
                for statement in 0..=data.statements.len() {
                    let location = Location { block, statement };
                    let mut used = Vec::new();
                    if let Some(it) = data.statements.get(statement) {
                        match &it.kind {
                            StatementKind::Assign(place, rvalue) => {
                                rvalue.for_each_operand(&mut |op| {
                                    if let OperandKind::Copy(p) | OperandKind::Move(p) = &op.kind {
                                        used.push(p.local);
                                    }
                                });
                                if let Rvalue::Ref(_, p)
                                | Rvalue::CopyForDeref(p)
                                | Rvalue::Len(p)
                                | Rvalue::Discriminant(p) = rvalue
                                {
                                    used.push(p.local);
                                }
                                if !place.projection.is_empty() {
                                    used.push(place.local);
                                }
                            }
                            _ => continue,
                        }
                    } else if let Some(terminator) = &data.terminator {
                        terminator.kind.for_each_operand(&mut |op| {
                            if let OperandKind::Copy(p) | OperandKind::Move(p) = &op.kind {
                                used.push(p.local);
                            }
                        });
                    }
                    for local in used {
                        if let Some(&id) = two_phase.get(&local)
                            && loans.loans[id].activation == Activation::NotActivated
                            && loans.loans[id].location != location
                        {
                            loans.loans[id].activation = Activation::At(location);
                            loans.activations.entry(location).or_default().push(id);
                        }
                    }
                }
            }
        }
        loans
    }

    /// Where each loan goes out of scope: the first points past it its region does not hold
    /// at, as rustc's `OutOfScopePrecomputer`.
    pub(super) fn compute_scopes(&mut self, body: &MirBody<'_>, cfg: &Cfg, values: &RegionValues) {
        for (id, loan) in self.loans.iter().enumerate() {
            let mut seen = BitSet::new(cfg.points);
            let mut work = successors_of(body, cfg, loan.location);
            while let Some(point) = work.pop() {
                if !seen.insert(point) {
                    continue;
                }
                if !values.contains_point(loan.region, point) {
                    self.out_of_scope.entry(point).or_default().push(id);
                    continue;
                }
                work.extend(successors_of(body, cfg, cfg.location(point)));
            }
        }
    }

    /// Polonius alpha's scopes: a loan goes out of scope at the first point after it is made
    /// that it is not live at, as rustc's `PoloniusOutOfScopePrecomputer`.
    pub(super) fn compute_polonius_scopes(
        &mut self,
        body: &MirBody<'_>,
        cfg: &Cfg,
        live: &[BitSet],
    ) {
        for (id, loan) in self.loans.iter().enumerate() {
            let issued = cfg.point(loan.location);
            let mut seen = BitSet::new(cfg.points);
            let mut work = successors_of(body, cfg, loan.location);
            while let Some(point) = work.pop() {
                if !seen.insert(point) {
                    continue;
                }
                if point != issued && !live[id].contains(point) {
                    self.out_of_scope.entry(point).or_default().push(id);
                    continue;
                }
                work.extend(successors_of(body, cfg, cfg.location(point)));
            }
        }
    }

    /// The loans in scope at the start of each block.
    pub(super) fn entry_sets<'db>(
        &self,
        ctx: &Ctx<'_, 'db>,
        body: &MirBody<'db>,
        cfg: &Cfg,
    ) -> Vec<BitSet> {
        let blocks = body.basic_blocks.len();
        let count = self.loans.len();
        let mut entry = vec![BitSet::new(count); blocks];
        let mut changed = true;
        while changed {
            changed = false;
            for &block in &cfg.order {
                let mut state = entry[index(block)].clone();
                let data = &body.basic_blocks[block];
                for statement in 0..=data.statements.len() {
                    let location = Location { block, statement };
                    self.before(cfg, location, &mut state);
                    self.after(ctx, body, location, &mut state);
                }
                for &successor in &cfg.successors[index(block)] {
                    changed |= entry[index(successor)].union(&state);
                }
            }
        }
        entry
    }

    /// The loans that go out of scope as `location` starts.
    pub(super) fn before(&self, cfg: &Cfg, location: Location, state: &mut BitSet) {
        if let Some(ids) = self.out_of_scope.get(&cfg.point(location)) {
            for &id in ids {
                state.remove(id);
            }
        }
    }

    /// What the statement or terminator at `location` does to the loans in scope.
    pub(super) fn after<'db>(
        &self,
        ctx: &Ctx<'_, 'db>,
        body: &MirBody<'db>,
        location: Location,
        state: &mut BitSet,
    ) {
        let data = &body.basic_blocks[location.block];
        if let Some(statement) = data.statements.get(location.statement) {
            match &statement.kind {
                StatementKind::Assign(place, _) => {
                    self.kill_on_place(ctx, body, place.local, place.projection.as_slice(), state);
                    if let Some(&id) = self.at.get(&location) {
                        state.insert(id);
                    }
                }
                StatementKind::StorageDead(local) => {
                    for &id in self.of_local.get(local).into_iter().flatten() {
                        state.remove(id);
                    }
                }
                StatementKind::InlineAsm(asm) => {
                    for operand in &asm.operands {
                        if let hir_ty::mir::AsmOperand::Reg { output: Some(output), .. } = operand {
                            self.kill_on_place(
                                ctx,
                                body,
                                output.local,
                                output.projection.as_slice(),
                                state,
                            );
                        }
                    }
                }
                _ => (),
            }
        } else if let Some(terminator) = &data.terminator {
            match &terminator.kind {
                TerminatorKind::Call { destination: place, .. }
                | TerminatorKind::DropAndReplace { place, .. } => {
                    self.kill_on_place(ctx, body, place.local, place.projection.as_slice(), state)
                }
                _ => (),
            }
        }
    }

    /// rustc's `kill_borrows_on_place`: overwriting a local ends every loan of what it holds.
    fn kill_on_place<'db>(
        &self,
        ctx: &Ctx<'_, 'db>,
        body: &MirBody<'db>,
        local: LocalId,
        projection: &[PlaceElem],
        state: &mut BitSet,
    ) {
        let Some(ids) = self.of_local.get(&local) else { return };
        for &id in ids {
            let loan = &self.loans[id];
            if projection.is_empty()
                || places_conflict(ctx, body, loan, local, projection, Depth::Deep, Bias::NoOverlap)
            {
                state.remove(id);
            }
        }
    }
}

fn successors_of(body: &MirBody<'_>, cfg: &Cfg, location: Location) -> Vec<usize> {
    let data = &body.basic_blocks[location.block];
    if location.statement < data.statements.len() {
        vec![cfg.point(Location { block: location.block, statement: location.statement + 1 })]
    } else {
        cfg.successors[index(location.block)]
            .iter()
            .map(|&it| cfg.point(Location { block: it, statement: 0 }))
            .collect()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Overlap {
    Arbitrary,
    EqualOrDisjoint,
    Disjoint,
}

/// Whether an access of `depth` to the place `local.projection` touches what `loan` borrows,
/// as rustc's `borrow_conflicts_with_place`.
pub(super) fn places_conflict<'db>(
    ctx: &Ctx<'_, 'db>,
    body: &MirBody<'db>,
    loan: &Loan,
    local: LocalId,
    projection: &[PlaceElem],
    depth: Depth,
    bias: Bias,
) -> bool {
    if loan.local != local {
        return false;
    }
    let borrow = loan.projection.as_slice();
    for (index, &elem) in borrow.iter().enumerate() {
        let base = &borrow[..index];
        let base_ty = ctx.place_ty(body, local, base);
        if let Some(&access) = projection.get(index) {
            match projection_conflict(ctx, base_ty, elem, access, bias) {
                Overlap::Arbitrary => return true,
                Overlap::EqualOrDisjoint => (),
                Overlap::Disjoint => return false,
            }
        } else {
            // The loan is of a part of the place accessed.
            match (elem, base_ty.ty.kind(), depth) {
                (_, _, Depth::ShallowArtificial) => return false,
                (ProjectionElem::Deref, _, Depth::Shallow) => return false,
                (
                    ProjectionElem::Deref,
                    TyKind::Ref(_, _, rustc_ast_ir::Mutability::Mut),
                    Depth::Drop,
                ) => {
                    return false;
                }
                (ProjectionElem::Deref, TyKind::Ref(_, _, rustc_ast_ir::Mutability::Not), _) => {
                    return false;
                }
                (ProjectionElem::Field(_), TyKind::Adt(def, _), Depth::Drop)
                    if base_ty.variant_id.is_none() && ctx.has_destructor(def.def_id()) =>
                {
                    return true;
                }
                _ => (),
            }
        }
    }
    // The access is of the place borrowed or of a part of it. A fake borrow of a place
    // does not conflict with an access of a part of it.
    !(loan.kind == BorrowKind::Shallow && borrow.len() < projection.len())
}

fn projection_conflict<'db>(
    ctx: &Ctx<'_, 'db>,
    base: hir_ty::mir::PlaceTy<'_>,
    borrow: PlaceElem,
    access: PlaceElem,
    bias: Bias,
) -> Overlap {
    let unknown = match bias {
        Bias::Overlap => Overlap::EqualOrDisjoint,
        Bias::NoOverlap => Overlap::Disjoint,
    };
    match (borrow, access) {
        (ProjectionElem::Deref, ProjectionElem::Deref) => Overlap::EqualOrDisjoint,
        (ProjectionElem::Field(a), ProjectionElem::Field(b)) => {
            if a == b {
                Overlap::EqualOrDisjoint
            } else {
                match base.ty.kind() {
                    TyKind::Adt(def, _) if def.is_union() => Overlap::Arbitrary,
                    _ => Overlap::Disjoint,
                }
            }
        }
        (ProjectionElem::Downcast(a), ProjectionElem::Downcast(b)) => {
            if a == b {
                Overlap::EqualOrDisjoint
            } else {
                Overlap::Disjoint
            }
        }
        (ProjectionElem::Index(..), ProjectionElem::Index(..))
        | (
            ProjectionElem::Index(..),
            ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. },
        )
        | (
            ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. },
            ProjectionElem::Index(..),
        ) => unknown,
        (
            ProjectionElem::ConstantIndex { offset: a, min_length: la, from_end: fa },
            ProjectionElem::ConstantIndex { offset: b, min_length: lb, from_end: fb },
        ) => {
            if fa == fb {
                if a == b { Overlap::EqualOrDisjoint } else { Overlap::Disjoint }
            } else if let Some(len) = array_len(ctx, base) {
                let (a, b) = (if fa { len - a } else { a }, if fb { len - b } else { b });
                if a == b { Overlap::EqualOrDisjoint } else { Overlap::Disjoint }
            } else {
                // Both patterns matched, so the slice has at least the longer minimum; an offset
                // from the end is then at least `min_length - offset` from the start.
                let (from_begin, from_end) = if fa { (b, a) } else { (a, b) };
                if from_begin >= la.max(lb) - from_end {
                    Overlap::EqualOrDisjoint
                } else {
                    Overlap::Disjoint
                }
            }
        }
        (
            ProjectionElem::ConstantIndex { offset, from_end, .. },
            ProjectionElem::Subslice { from, to },
        )
        | (
            ProjectionElem::Subslice { from, to },
            ProjectionElem::ConstantIndex { offset, from_end, .. },
        ) => {
            let overlaps = match (array_len(ctx, base), from_end) {
                (Some(len), _) => {
                    let index = if from_end { len.wrapping_sub(offset) } else { offset };
                    index >= from && index < len.saturating_sub(to)
                }
                (None, false) => offset >= from,
                (None, true) => offset > to,
            };
            if overlaps { Overlap::EqualOrDisjoint } else { Overlap::Disjoint }
        }
        (
            ProjectionElem::Subslice { from: fa, to: ta },
            ProjectionElem::Subslice { from: fb, to: tb },
        ) => match array_len(ctx, base) {
            Some(len) if fa.max(fb) >= len.saturating_sub(ta.max(tb)) => Overlap::Disjoint,
            _ => Overlap::EqualOrDisjoint,
        },
        _ => Overlap::Arbitrary,
    }
}

fn array_len(ctx: &Ctx<'_, '_>, base: hir_ty::mir::PlaceTy<'_>) -> Option<u64> {
    match base.ty.kind() {
        TyKind::Array(_, len) => {
            hir_ty::consteval::try_const_usize(ctx.db, len).map(|it| it as u64)
        }
        _ => None,
    }
}
