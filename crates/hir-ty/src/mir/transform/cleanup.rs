//! The ways out of a body that unwinds: every call and drop names the cleanup block it goes
//! to, as rustc's MIR building makes them, and statements that panic go to
//! [`MirBody::panic_cleanup`].
//!
//! rustc builds a chain of cleanup blocks per scope, each dropping what that scope holds. Here
//! there is one chain for the whole body, which drops every local that may need dropping in
//! the order a landing pad of rustc would. Drop elaboration then decides from where the body
//! can unwind which of those drops run and which test a flag. The order is the one rustd's
//! code generator had for its single landing pad.

use rustc_hash::{FxHashMap, FxHashSet};

use crate::mir::{
    BasicBlock, BasicBlockId, LocalId, MirBody, MirSpan, Operand, OperandKind, Place,
    ProjectionElem, Rvalue, Statement, StatementKind, StoredPlace, Terminator, TerminatorKind,
    return_slot,
};

use super::Ctx;

/// Moves what a terminator reads at a runtime index into a temporary first, in a statement,
/// and has a call write a place that is indexed through a temporary and an assignment after
/// it returns. A terminator then unwinds only from what it calls, which rustc's MIR has too:
/// its index checks are `Assert`s of their own.
pub(super) fn hoist_indexing<'db>(ctx: &Ctx<'db>, body: &mut MirBody<'db>) {
    let indexes = |place: &StoredPlace| {
        place.projection.as_slice().iter().any(|it| matches!(it, ProjectionElem::Index(_)))
    };
    let ids: Vec<BasicBlockId> = body.basic_blocks.iter().map(|(id, _)| id).collect();
    for id in ids {
        let Some(terminator) = body.basic_blocks[id].terminator.take() else { continue };
        let span = terminator.span;
        let mut kind = terminator.kind;
        let mut hoisted = Vec::new();
        kind.for_each_operand_mut(&mut |operand| {
            let (OperandKind::Copy(place) | OperandKind::Move(place)) = &operand.kind else {
                return;
            };
            if !indexes(place) {
                return;
            }
            let ty = place.as_ref().ty(body, &ctx.infcx, ctx.env).ty;
            let temp = body.locals.alloc(crate::mir::Local { ty: ty.store() });
            let read = Operand { kind: operand.kind.clone(), span: operand.span };
            hoisted.push(Statement {
                kind: StatementKind::Assign(Place::from(temp).store(), Rvalue::Use(read)),
                span,
            });
            operand.kind = match operand.kind {
                OperandKind::Move(_) => OperandKind::Move(Place::from(temp).store()),
                _ => OperandKind::Copy(Place::from(temp).store()),
            };
        });
        match &mut kind {
            TerminatorKind::Call { destination, target: Some(target), .. }
                if indexes(destination) =>
            {
                let ty = destination.as_ref().ty(body, &ctx.infcx, ctx.env).ty;
                let temp = body.locals.alloc(crate::mir::Local { ty: ty.store() });
                let place = std::mem::replace(destination, Place::from(temp).store());
                let is_copy = ctx.infcx.type_is_copy_modulo_regions(ctx.env, ty);
                let value = match is_copy {
                    true => OperandKind::Copy(Place::from(temp).store()),
                    false => OperandKind::Move(Place::from(temp).store()),
                };
                let write = body.basic_blocks.alloc(BasicBlock {
                    statements: vec![Statement {
                        kind: StatementKind::Assign(
                            place,
                            Rvalue::Use(Operand { kind: value, span: None }),
                        ),
                        span,
                    }],
                    terminator: Some(Terminator {
                        span,
                        kind: TerminatorKind::Goto { target: *target },
                    }),
                    is_cleanup: false,
                });
                *target = write;
            }
            // The place is found, and its index checked, before what it holds is dropped.
            TerminatorKind::Drop { place, .. } if indexes(place) => {
                hoisted.push(Statement { kind: StatementKind::FakeRead(place.clone()), span });
            }
            _ => (),
        }
        let block = &mut body.basic_blocks[id];
        block.statements.extend(hoisted);
        block.terminator = Some(Terminator { span, kind });
    }
}

/// Gives every call and drop that is not in a cleanup block the cleanup chain to unwind to, and
/// a drop that `place = value` makes before it writes a cleanup block that writes all the
/// same, so that the place is not dropped again on the way out.
pub(super) fn add_cleanup_paths<'db>(ctx: &Ctx<'db>, body: &mut MirBody<'db>) {
    let order = unwinding_order(ctx, body);
    if order.is_empty() {
        return;
    }
    let ends = scope_ends(body);
    let mut next = body.basic_blocks.alloc(BasicBlock {
        statements: Vec::new(),
        terminator: Some(Terminator { span: MirSpan::Unknown, kind: TerminatorKind::UnwindResume }),
        is_cleanup: true,
    });
    for &local in order.iter().rev() {
        let span = ends.get(&local).copied().unwrap_or(MirSpan::Unknown);
        next = body.basic_blocks.alloc(BasicBlock {
            statements: Vec::new(),
            terminator: Some(Terminator {
                span,
                kind: TerminatorKind::Drop {
                    place: Place::from(local).store(),
                    target: next,
                    unwind: None,
                },
            }),
            is_cleanup: true,
        });
    }
    let chain = next;
    body.panic_cleanup = Some(chain);
    let ids: Vec<BasicBlockId> = body.basic_blocks.iter().map(|(id, _)| id).collect();
    for id in ids {
        let block = &body.basic_blocks[id];
        if block.is_cleanup {
            continue;
        }
        let Some(terminator) = &block.terminator else { continue };
        let replaced = match &terminator.kind {
            TerminatorKind::Drop { place, target, .. }
                if !place.projection.as_slice().is_empty() =>
            {
                body.basic_blocks[*target]
                    .statements
                    .iter()
                    .find_map(|it| match &it.kind {
                        StatementKind::Assign(to, rvalue) => Some((to, rvalue)),
                        _ => None,
                    })
                    .filter(|(to, _)| *to == place)
                    .map(|(to, rvalue)| (to.clone(), rvalue.clone()))
            }
            _ => None,
        };
        let span = terminator.span;
        let unwind = match replaced {
            Some((to, rvalue)) => body.basic_blocks.alloc(BasicBlock {
                statements: vec![Statement { kind: StatementKind::Assign(to, rvalue), span }],
                terminator: Some(Terminator { span, kind: TerminatorKind::Goto { target: chain } }),
                is_cleanup: true,
            }),
            None => chain,
        };
        let terminator = body.basic_blocks[id].terminator.as_mut().expect("a terminator");
        if let Some(slot @ None) = terminator.kind.unwind_mut() {
            *slot = Some(unwind);
        }
    }
}

/// Where the scope of each local that the body drops ends.
fn scope_ends(body: &MirBody<'_>) -> FxHashMap<LocalId, MirSpan> {
    let mut ends = FxHashMap::default();
    for (_, block) in body.basic_blocks.iter() {
        if let Some(Terminator {
            kind: TerminatorKind::Drop { place, .. },
            span: span @ MirSpan::ExprEnd(_),
        }) = &block.terminator
        {
            ends.entry(place.local).or_insert(*span);
        }
    }
    ends
}

/// The locals that may need dropping, in the order a landing pad drops those that hold a value.
fn unwinding_order<'db>(ctx: &Ctx<'db>, body: &MirBody<'db>) -> Vec<LocalId> {
    let flagged: Vec<LocalId> = body
        .locals
        .iter()
        .filter(|&(local, decl)| local != return_slot() && ctx.needs_drop(decl.ty.as_ref()))
        .map(|(local, _)| local)
        .collect();
    if flagged.is_empty() {
        return Vec::new();
    }
    let flagged_set: FxHashSet<LocalId> = flagged.iter().copied().collect();
    let drop_order = drop_order(body);
    let operand_temps = operand_temps(body);
    let named: FxHashSet<LocalId> =
        body.binding_locals.values().chain(&body.param_locals).copied().collect();
    let mut undropped: Vec<LocalId> =
        flagged.iter().copied().filter(|local| !drop_order.contains(local)).collect();
    undropped.reverse();
    // Temporaries the body never drops are moved by the expression that made them, so one
    // that still holds a value belongs to the expression that unwound: its operands come
    // first, before the temporaries it borrowed, which live to the end of the statement.
    // The rest go in the order the body drops them when nothing unwinds, which is innermost
    // scope first.
    let (variables, mut order): (Vec<LocalId>, Vec<LocalId>) =
        undropped.into_iter().partition(|local| named.contains(local));
    order.sort_by_key(|local| !operand_temps.contains(local));
    order.extend(drop_order.iter().copied().filter(|it| flagged_set.contains(it)));
    // A variable the body never drops is in a scope no path leaves by its end. It goes
    // after what its scope holds and before the variables declared ahead of it.
    for variable in variables {
        let outer = order.iter().position(|it| named.contains(it) && *it < variable);
        order.insert(outer.unwrap_or(order.len()), variable);
    }
    order
}

/// The locals that the body drops as a whole, in the order the scopes holding them end.
fn drop_order(body: &MirBody<'_>) -> Vec<LocalId> {
    let successors = |id: BasicBlockId| -> Vec<BasicBlockId> {
        body.basic_blocks[id]
            .terminator
            .as_ref()
            .map(|it| it.kind.successors().into_vec())
            .unwrap_or_default()
    };
    let mut seen = FxHashSet::default();
    let mut post_order = Vec::new();
    let mut stack = vec![(body.start_block, successors(body.start_block))];
    seen.insert(body.start_block);
    while let Some((id, pending)) = stack.last_mut() {
        match pending.pop() {
            Some(next) if seen.insert(next) => {
                let more = successors(next);
                stack.push((next, more));
            }
            Some(_) => (),
            None => {
                post_order.push(*id);
                stack.pop();
            }
        }
    }
    // `local = value` drops what the local held first, which says nothing about its scope, so
    // those drops only count for a local that has no other.
    let replaces = |place: &StoredPlace, target: BasicBlockId| {
        let assigned = body.basic_blocks[target].statements.iter().find_map(|it| match &it.kind {
            StatementKind::Assign(to, _) => Some(to),
            _ => None,
        });
        assigned == Some(place)
    };
    let mut order: Vec<LocalId> = Vec::new();
    for replaced in [false, true] {
        for &id in post_order.iter().rev() {
            if let Some(TerminatorKind::Drop { place, target, .. }) =
                body.basic_blocks[id].terminator.as_ref().map(|it| &it.kind)
                && place.projection.as_slice().is_empty()
                && replaces(place, *target) == replaced
                && !order.contains(&place.local)
            {
                order.push(place.local);
            }
        }
    }
    // The scopes say which of two locals is dropped first. Where the blocks come in the body
    // does not: the exit of a loop can come before its body. A parameter is in no scope and
    // outlives them all.
    let ends: FxHashMap<LocalId, usize> =
        body.scope_ends.iter().enumerate().map(|(at, &local)| (local, at)).collect();
    order.sort_by_key(|local| ends.get(local).copied().unwrap_or(ends.len()));
    order
}

/// The temporaries that no place of the body holds on to: the ones an expression made for its
/// operands.
fn operand_temps(body: &MirBody<'_>) -> FxHashSet<LocalId> {
    let mut temps: FxHashSet<LocalId> = body.locals.iter().map(|(id, _)| id).collect();
    for &local in body.binding_locals.values().chain(&body.param_locals) {
        temps.remove(&local);
    }
    let mut held = |place: &StoredPlace| {
        temps.remove(&place.local);
    };
    let operand = |operand: &Operand, held: &mut dyn FnMut(&StoredPlace)| {
        if let OperandKind::Copy(place) | OperandKind::Move(place) = &operand.kind
            && !place.projection.as_slice().is_empty()
        {
            held(place);
        }
    };
    for (_, block) in body.basic_blocks.iter() {
        for statement in &block.statements {
            let StatementKind::Assign(to, rvalue) = &statement.kind else { continue };
            if !to.projection.as_slice().is_empty() {
                held(to);
            }
            match rvalue {
                Rvalue::Ref(_, place)
                | Rvalue::CopyForDeref(place)
                | Rvalue::Discriminant(place)
                | Rvalue::Len(place) => held(place),
                Rvalue::Use(it) | Rvalue::UnaryOp(_, it) | Rvalue::Cast(_, it, _) => {
                    operand(it, &mut held)
                }
                Rvalue::Aggregate(_, operands) => {
                    operands.iter().for_each(|it| operand(it, &mut held))
                }
                _ => (),
            }
        }
        if let Some(TerminatorKind::Call { args, .. }) =
            block.terminator.as_ref().map(|it| &it.kind)
        {
            args.iter().for_each(|it| operand(it, &mut held));
        }
    }
    temps
}
