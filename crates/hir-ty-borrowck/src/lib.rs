//! The borrow checker: rustc's `rustc_borrowck` over the bodies MIR lowering builds.
//!
//! Each body is checked on its own, closures and coroutines with it, as rustc checks a body
//! and the closures in it together, and the result is a query of the body's owner, so an edit checks again
//! only the bodies it changes. What it reports carries rustc's error codes and the span rustc
//! points at; `docs/borrowck.md` in the repository describes what is checked and how.

extern crate ra_ap_rustc_ast_ir as rustc_ast_ir;
extern crate ra_ap_rustc_type_ir as rustc_type_ir;

mod bitset;
mod cfg;
mod check;
mod liveness;
mod loans;
mod moves;
mod names;
mod polonius;
mod regions;

use std::sync::LazyLock;

use base_db::Crate;
use hir_def::{
    DefWithBodyId, HasModule,
    expr_store::ExpressionStore,
    hir::{BindingAnnotation, BindingId, ExprId},
};
use rustc_hash::FxHashMap;
use salsa::SalsaValue;

use hir_ty::{
    InferBodyId, InferenceResult,
    closure_analysis::CapturedPlace,
    db::{HirDatabase, InternedCoroutineId},
    next_solver::{
        DbInterner, ParamEnv, Ty, TypingMode,
        infer::{DbInternerInferExt, InferCtxt},
    },
};

use hir_ty::mir::{
    LocalId, MirBody, MirSpan, Operand, OperandKind, PlaceElem, PlaceTy, StoredPlace,
};

/// Which of rustc's borrow checkers to match: the one stable Rust has, or the location
/// sensitive one nightly Rust has by default (`-Zpolonius=next`), which accepts more.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Nll,
    Polonius,
    Off,
}

/// `RA_JIT_BORROWCK`: `polonius` (the default), `nll` or `off`.
pub fn mode() -> Mode {
    static MODE: LazyLock<Mode> =
        LazyLock::new(|| match std::env::var("RA_JIT_BORROWCK").as_deref() {
            Ok("nll") => Mode::Nll,
            Ok("off" | "0") => Mode::Off,
            _ => Mode::Polonius,
        });
    *MODE
}

#[derive(Debug, Clone, PartialEq, Eq, SalsaValue)]
pub struct BorrowckDiagnostic {
    /// rustc's error code, as `E0382`.
    pub code: &'static str,
    pub span: MirSpan,
    pub message: String,
    /// Reported at the function's signature rather than at `span`, as rustc reports an
    /// `impl Trait`'s hidden type.
    pub at_signature: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, SalsaValue)]
pub struct BorrowckResult {
    pub diagnostics: Vec<BorrowckDiagnostic>,
    /// Why the body, or a closure in it, was not checked: MIR lowering does not support
    /// something in it.
    pub skipped: Option<String>,
}

/// The errors rustc's borrow checker reports in `def` and the closures in it.
#[salsa::tracked(returns(ref))]
pub fn borrowck<'db>(db: &'db dyn HirDatabase, def: DefWithBodyId) -> BorrowckResult {
    let mode = mode();
    let mut result = BorrowckResult::default();
    if mode == Mode::Off {
        return result;
    }
    let owner = InferBodyId::DefWithBodyId(def);
    let infer = InferenceResult::of(db, owner);
    if infer.has_type_mismatches() || infer.is_erroneous() {
        return result;
    }
    let body = match db.mir_body(owner) {
        Ok(body) => body,
        Err(error) => {
            result.skipped = Some(format!("{error:?}"));
            return result;
        }
    };
    let store = ExpressionStore::of(db, owner.expression_store_owner(db));
    let mut bodies = vec![(body, None)];
    let mut index = 0;
    while let Some(&(body, _)) = bodies.get(index) {
        index += 1;
        for &closure in &body.closures {
            match db.mir_body_for_closure(closure) {
                Ok(it) => bodies.push((it, Some(closure.loc(db).expr))),
                Err(error) => {
                    result.skipped.get_or_insert_with(|| format!("closure: {error:?}"));
                }
            }
        }
        // The body of an `async` block or `async fn`, checked as rustc checks a coroutine: as
        // a closure of the body that makes it.
        for coroutine in coroutines(body) {
            match db.mir_body_for_coroutine(coroutine) {
                Ok(it) => bodies.push((it, Some(coroutine.loc(db).expr))),
                Err(error) => {
                    result.skipped.get_or_insert_with(|| format!("coroutine: {error:?}"));
                }
            }
        }
    }
    // A closure after the closures in it and before the body that makes it, which proves what
    // the closure needs of it.
    let mut requirements = FxHashMap::default();
    let mut diagnostics = Vec::new();
    for (body, closure) in bodies.into_iter().rev() {
        let ctx = Ctx::new(db, owner, store, infer, body, closure, mode);
        let mut found = Vec::new();
        let needs = check::check(&ctx, body, &requirements, &mut found);
        diagnostics.push(found);
        if let Some(closure) = closure {
            requirements.insert(closure, needs);
        }
    }
    result.diagnostics.extend(diagnostics.into_iter().rev().flatten());
    result
}

/// The coroutines `body` makes, in the order it makes them.
fn coroutines<'db>(body: &MirBody<'db>) -> Vec<InternedCoroutineId<'db>> {
    use hir_ty::mir::{AggregateKind, Rvalue, StatementKind};
    use hir_ty::next_solver::TyKind;
    use rustc_type_ir::inherent::IntoKind;
    let mut out = Vec::new();
    for (_, block) in body.basic_blocks.iter() {
        for statement in &block.statements {
            if let StatementKind::Assign(_, Rvalue::Aggregate(AggregateKind::Closure(ty), _)) =
                &statement.kind
                && let TyKind::Coroutine(id, _) = ty.as_ref().kind()
                && !out.contains(&id.0)
            {
                out.push(id.0);
            }
        }
    }
    out
}

/// What the checks ask of the types and the source of a body.
pub(crate) struct Ctx<'a, 'db> {
    pub(crate) db: &'db dyn HirDatabase,
    pub(crate) owner: InferBodyId<'db>,
    pub(crate) infcx: InferCtxt<'db>,
    pub(crate) env: ParamEnv<'db>,
    #[allow(dead_code)]
    pub(crate) krate: Crate,
    pub(crate) store: &'a ExpressionStore,
    #[allow(dead_code)]
    pub(crate) infer: &'a InferenceResult<'db>,
    #[allow(dead_code)]
    pub(crate) mode: Mode,
    /// The user variable each local is, if it is one.
    pub(crate) bindings: FxHashMap<LocalId, BindingId>,
    /// In the body of a closure, its captures, by field of the closure.
    pub(crate) upvars: Vec<Upvar>,
    /// In the body of a closure, how it is called.
    pub(crate) closure_kind: Option<rustc_type_ir::ClosureKind>,
    pub(crate) closure_local: Option<LocalId>,
}

/// A capture of a closure, in its body.
pub(crate) struct Upvar {
    pub(crate) binding: BindingId,
    /// The local lowering copies the capture into.
    pub(crate) local: LocalId,
    pub(crate) by_ref: bool,
    pub(crate) mutable: bool,
    /// The captured place is behind a reference.
    pub(crate) behind_deref: bool,
    #[allow(dead_code)]
    pub(crate) place: CapturedPlace,
}

impl<'a, 'db> Ctx<'a, 'db> {
    fn new(
        db: &'db dyn HirDatabase,
        owner: InferBodyId<'db>,
        store: &'a ExpressionStore,
        infer: &'a InferenceResult<'db>,
        body: &MirBody<'db>,
        closure: Option<ExprId>,
        mode: Mode,
    ) -> Self {
        let krate = owner.krate(db);
        let env = db.trait_environment(owner.generic_def(db));
        let interner = DbInterner::new_with(db, krate);
        let infcx = interner.infer_ctxt().build(TypingMode::non_body_analysis());
        let bindings =
            body.binding_locals.iter().map(|(binding, &local)| (local, binding)).collect();
        let mut upvars = Vec::new();
        let mut closure_kind = None;
        let mut closure_local = None;
        if let Some(expr) = closure {
            closure_kind = closure_kind_of(infer, expr);
            closure_local = body.param_locals.first().copied();
            if let Some(data) = infer.closures_data.get(&expr) {
                let mut locals: Vec<LocalId> =
                    body.upvar_locals.values().flatten().map(|(local, _)| *local).collect();
                locals.sort_by_key(|it| it.into_raw().into_u32());
                for (capture, local) in data.min_captures.values().flatten().zip(locals) {
                    upvars.push(Upvar {
                        binding: capture.captured_local(),
                        local,
                        by_ref: capture.is_by_ref(),
                        mutable: capture.mutability == rustc_ast_ir::Mutability::Mut,
                        behind_deref: capture.place.projections.iter().any(|it| {
                            matches!(it.kind, hir_ty::closure_analysis::ProjectionKind::Deref)
                        }),
                        place: capture.clone(),
                    });
                }
            }
        }
        Ctx {
            db,
            owner,
            infcx,
            env,
            krate,
            store,
            infer,
            mode,
            bindings,
            upvars,
            closure_kind,
            closure_local,
        }
    }

    pub(crate) fn place_ty(
        &self,
        body: &MirBody<'db>,
        local: LocalId,
        projection: &[PlaceElem],
    ) -> PlaceTy<'db> {
        PlaceTy::from_ty(body.locals[local].ty.as_ref()).multi_projection_ty(
            &self.infcx,
            self.env,
            projection,
        )
    }

    pub(crate) fn project(&self, ty: PlaceTy<'db>, elem: &PlaceElem) -> PlaceTy<'db> {
        ty.projection_ty(&self.infcx, elem, self.env)
    }

    /// Whether a value of `ty` neither needs dropping nor has interior mutability, which a
    /// constant must to be promoted.
    pub(crate) fn is_plain(&self, ty: Ty<'db>) -> bool {
        hir_ty::drop::has_drop_glue(&self.infcx, ty, self.env) == hir_ty::drop::DropGlue::None
            && hir_ty::drop::is_freeze(self.db, self.infcx.interner, ty)
    }

    pub(crate) fn has_destructor(&self, adt: hir_def::AdtId) -> bool {
        hir_ty::drop::destructor(self.db, adt).is_some()
    }

    pub(crate) fn is_copy(&self, ty: Ty<'db>) -> bool {
        self.infcx.type_is_copy_modulo_regions(self.env, ty)
    }

    /// The place `operand` moves out of. Lowering says a value a closure captures by value is
    /// moved even when its type is `Copy`.
    pub(crate) fn moved_place<'o>(
        &self,
        body: &MirBody<'db>,
        operand: &'o Operand,
    ) -> Option<&'o StoredPlace> {
        let OperandKind::Move(place) = &operand.kind else { return None };
        let ty = self.place_ty(body, place.local, place.projection.as_slice()).ty;
        (!self.is_copy(ty)).then_some(place)
    }

    /// Whether `local` is a variable declared without `mut`, which can only be assigned once.
    pub(crate) fn is_immutable(&self, local: LocalId) -> bool {
        if self.upvars.iter().any(|it| it.local == local) {
            return false;
        }
        match self.bindings.get(&local) {
            Some(&binding) => self.store[binding].mode != BindingAnnotation::Mutable,
            None => false,
        }
    }

    pub(crate) fn upvar_of_local(&self, local: LocalId) -> Option<usize> {
        self.upvars.iter().position(|it| it.local == local)
    }
}

fn closure_kind_of(
    infer: &InferenceResult<'_>,
    expr: ExprId,
) -> Option<rustc_type_ir::ClosureKind> {
    use rustc_type_ir::inherent::{GenericArgs as _, IntoKind};
    match infer.expr_ty(expr).kind() {
        hir_ty::next_solver::TyKind::Closure(_, args) => Some(args.as_closure().kind()),
        hir_ty::next_solver::TyKind::CoroutineClosure(_, args) => {
            Some(args.as_coroutine_closure().kind())
        }
        _ => None,
    }
}

/// The local a body returns in, as `hir_ty::mir`'s own private `return_slot`.
pub(crate) fn return_slot() -> LocalId {
    LocalId::from_raw(la_arena::RawIdx::from(0))
}

/// Whether `rvalue` compares, which takes its operands by reference whatever their type:
/// rust-analyzer lowers a constant pattern of a type that is not `Copy` to `==` on the value.
pub(crate) fn compares(rvalue: &hir_ty::mir::Rvalue) -> bool {
    use hir_ty::mir::{BinOp, Rvalue};
    match rvalue {
        Rvalue::BinaryOp(op, ..) | Rvalue::CheckedBinaryOp(op, ..) => {
            matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge)
        }
        _ => false,
    }
}
