//! The passes that take a body from [`MirPhase::Built`] to [`MirPhase::Runtime`], as rustc's
//! `run_runtime_lowering_passes` does, in the order they run:
//!
//! 1. [`cleanup::hoist_indexing`]: a terminator unwinds only from what it calls.
//! 2. [`cleanup::add_cleanup_paths`]: every call and drop names where it unwinds to.
//! 3. [`elaborate_drops::elaborate_drops`]: every drop drops; drop flags are locals.
//! 4. [`packed_drops::add_moves_for_packed_drops`]: what a packed struct leaves misaligned is
//!    moved out before it is dropped.
//!
//! None of them optimizes. The IDE never asks for these queries, so it never pays for them;
//! the code generator and the interpreter do. `docs/mir-passes.md` has the design.

mod cleanup;
mod elaborate_drops;
mod move_paths;
mod packed_drops;

use std::cell::RefCell;

use base_db::Crate;
use hir_def::HasModule;
use rustc_hash::FxHashMap;
use rustc_type_ir::inherent::{IntoKind, Ty as _};

use crate::{
    InferBodyId,
    db::{HirDatabase, InternedClosureId, InternedCoroutineId},
    drop::{DropGlue, has_drop_glue},
    next_solver::{
        DbInterner, ErrorGuaranteed, ParamEnv, StoredGenericArgs, Ty, TyKind, TypingMode,
        infer::{DbInternerInferExt, InferCtxt, traits::ObligationCause},
        obligation_ctxt::ObligationCtxt,
    },
    traits::{ParamEnvAndCrate, StoredParamEnvAndCrate},
};

use super::{MirBody, MirLowerError, MirPhase, monomorphization::monomorphized};

type Result<'db, T> = std::result::Result<T, MirLowerError<'db>>;

/// What the passes ask of the types of a body.
pub(super) struct Ctx<'db> {
    db: &'db dyn HirDatabase,
    infcx: InferCtxt<'db>,
    env: ParamEnv<'db>,
    krate: Crate,
    needs_drop: RefCell<FxHashMap<Ty<'db>, bool>>,
}

impl<'db> Ctx<'db> {
    fn new(db: &'db dyn HirDatabase, owner: InferBodyId<'db>) -> Self {
        let krate = owner.krate(db);
        let env = db.trait_environment(owner.generic_def(db));
        let interner = DbInterner::new_with(db, krate);
        let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());
        Ctx { db, infcx, env, krate, needs_drop: RefCell::default() }
    }

    /// Whether a value of `ty` may have something to drop: a type parameter may.
    fn needs_drop(&self, ty: Ty<'db>) -> bool {
        if let Some(&known) = self.needs_drop.borrow().get(&ty) {
            return known;
        }
        let needs = has_drop_glue(&self.infcx, ty, self.env) != DropGlue::None;
        self.needs_drop.borrow_mut().insert(ty, needs);
        needs
    }

    fn normalize(&self, ty: Ty<'db>) -> Ty<'db> {
        if !matches!(ty.kind(), TyKind::Alias(..)) {
            return ty;
        }
        let mut ocx = ObligationCtxt::new(&self.infcx);
        ocx.structurally_normalize_ty(&ObligationCause::dummy(), self.env, ty)
            .unwrap_or_else(|_| Ty::new_error(self.infcx.interner, ErrorGuaranteed))
    }

    fn has_destructor(&self, adt: hir_def::AdtId) -> bool {
        crate::drop::destructor(self.db, adt).is_some()
    }

    fn bool_ty(&self) -> Ty<'db> {
        Ty::new_bool(self.infcx.interner)
    }

    /// The alignment of `ty`, if its layout is known where the body is generic.
    fn layout_align(&self, ty: Ty<'db>) -> Option<u64> {
        let env = ParamEnvAndCrate { param_env: self.env, krate: self.krate };
        let layout = self.db.layout_of_ty(ty.store(), env.store()).ok()?;
        Some(layout.align.abi.bytes())
    }
}

/// `built` after the passes.
fn run_passes<'db>(db: &'db dyn HirDatabase, built: &MirBody<'db>) -> Result<'db, MirBody<'db>> {
    let mut body = built.clone();
    // The body of an `async` block keeps what it holds across a yield in the future, with the
    // flags the code generator makes for it: it is not elaborated yet.
    if body.coroutine.is_some() {
        return Ok(body);
    }
    let ctx = Ctx::new(db, body.owner);
    cleanup::hoist_indexing(&ctx, &mut body);
    cleanup::add_cleanup_paths(&ctx, &mut body);
    elaborate_drops::elaborate_drops(&ctx, &mut body)?;
    packed_drops::add_moves_for_packed_drops(&ctx, &mut body);
    body.phase = MirPhase::Runtime;
    body.shrink_to_fit();
    Ok(body)
}

#[salsa::tracked(returns(as_ref), cycle_result = runtime_mir_body_cycle_result)]
pub fn runtime_mir_body_query<'db>(
    db: &'db dyn HirDatabase,
    def: InferBodyId<'db>,
) -> Result<'db, MirBody<'db>> {
    run_passes(db, db.mir_body(def)?)
}

/// The constant behind a promoted `&expr` of `def` after the passes: what the code generator
/// evaluates for it with the interpreter.
#[salsa::tracked(returns(as_ref))]
pub fn runtime_promoted_mir_body_query<'db>(
    db: &'db dyn HirDatabase,
    def: InferBodyId<'db>,
    expr: hir_def::hir::ExprId,
) -> Result<'db, MirBody<'db>> {
    let built = super::promoted_mir_body_query(db, def, expr).map_err(Clone::clone)?;
    run_passes(db, built)
}

#[salsa::tracked(returns(as_ref), cycle_result = runtime_mir_body_for_closure_cycle_result)]
pub fn runtime_mir_body_for_closure_query<'db>(
    db: &'db dyn HirDatabase,
    closure: InternedClosureId<'db>,
) -> Result<'db, MirBody<'db>> {
    run_passes(db, db.mir_body_for_closure(closure)?)
}

#[salsa::tracked(returns(as_ref), cycle_result = runtime_mir_body_for_coroutine_cycle_result)]
pub fn runtime_mir_body_for_coroutine_query<'db>(
    db: &'db dyn HirDatabase,
    coroutine: InternedCoroutineId<'db>,
) -> Result<'db, MirBody<'db>> {
    run_passes(db, db.mir_body_for_coroutine(coroutine)?)
}

#[salsa::tracked(returns(as_ref), cycle_result = runtime_mir_body_for_coroutine_cycle_result)]
pub fn runtime_mir_body_for_coroutine_by_move_query<'db>(
    db: &'db dyn HirDatabase,
    coroutine: InternedCoroutineId<'db>,
) -> Result<'db, MirBody<'db>> {
    run_passes(db, db.mir_body_for_coroutine_by_move(coroutine)?)
}

#[salsa::tracked(returns(as_ref), cycle_result = monomorphized_runtime_mir_body_cycle_result)]
pub fn monomorphized_runtime_mir_body_query<'db>(
    db: &'db dyn HirDatabase,
    def: InferBodyId<'db>,
    subst: StoredGenericArgs,
    trait_env: StoredParamEnvAndCrate,
) -> Result<'db, MirBody<'db>> {
    monomorphized(db, db.runtime_mir_body(def)?, &subst, &trait_env)
}

#[salsa::tracked(
    returns(as_ref),
    cycle_result = monomorphized_runtime_mir_body_for_closure_cycle_result
)]
pub fn monomorphized_runtime_mir_body_for_closure_query<'db>(
    db: &'db dyn HirDatabase,
    closure: InternedClosureId<'db>,
    subst: StoredGenericArgs,
    trait_env: StoredParamEnvAndCrate,
) -> Result<'db, MirBody<'db>> {
    monomorphized(db, db.runtime_mir_body_for_closure(closure)?, &subst, &trait_env)
}

#[salsa::tracked(
    returns(as_ref),
    cycle_result = monomorphized_runtime_mir_body_for_coroutine_cycle_result
)]
pub fn monomorphized_runtime_mir_body_for_coroutine_query<'db>(
    db: &'db dyn HirDatabase,
    coroutine: InternedCoroutineId<'db>,
    subst: StoredGenericArgs,
    trait_env: StoredParamEnvAndCrate,
) -> Result<'db, MirBody<'db>> {
    monomorphized(db, db.runtime_mir_body_for_coroutine(coroutine)?, &subst, &trait_env)
}

#[salsa::tracked(
    returns(as_ref),
    cycle_result = monomorphized_runtime_mir_body_for_coroutine_cycle_result
)]
pub fn monomorphized_runtime_mir_body_for_coroutine_by_move_query<'db>(
    db: &'db dyn HirDatabase,
    coroutine: InternedCoroutineId<'db>,
    subst: StoredGenericArgs,
    trait_env: StoredParamEnvAndCrate,
) -> Result<'db, MirBody<'db>> {
    monomorphized(db, db.runtime_mir_body_for_coroutine_by_move(coroutine)?, &subst, &trait_env)
}

fn runtime_mir_body_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: salsa::Id,
    _: InferBodyId<'db>,
) -> Result<'db, MirBody<'db>> {
    Err(MirLowerError::Loop)
}

fn runtime_mir_body_for_closure_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: salsa::Id,
    _: InternedClosureId<'db>,
) -> Result<'db, MirBody<'db>> {
    Err(MirLowerError::Loop)
}

fn runtime_mir_body_for_coroutine_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: salsa::Id,
    _: InternedCoroutineId<'db>,
) -> Result<'db, MirBody<'db>> {
    Err(MirLowerError::Loop)
}

fn monomorphized_runtime_mir_body_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: salsa::Id,
    _: InferBodyId<'db>,
    _: StoredGenericArgs,
    _: StoredParamEnvAndCrate,
) -> Result<'db, MirBody<'db>> {
    Err(MirLowerError::Loop)
}

fn monomorphized_runtime_mir_body_for_closure_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: salsa::Id,
    _: InternedClosureId<'db>,
    _: StoredGenericArgs,
    _: StoredParamEnvAndCrate,
) -> Result<'db, MirBody<'db>> {
    Err(MirLowerError::Loop)
}

fn monomorphized_runtime_mir_body_for_coroutine_cycle_result<'db>(
    _db: &'db dyn HirDatabase,
    _: salsa::Id,
    _: InternedCoroutineId<'db>,
    _: StoredGenericArgs,
    _: StoredParamEnvAndCrate,
) -> Result<'db, MirBody<'db>> {
    Err(MirLowerError::Loop)
}
