# MIR passes: from built MIR to runtime MIR

This fork (indexable-inc/rust-analyzer, branch `ra-jit`) gives rust-analyzer's MIR the
pre-codegen passes rustc runs, so that the code generator of
[ra-jit](https://github.com/indexable-inc/ra-jit) (`crates/hir-ty-codegen`) and the interpreter
(`hir-ty/src/mir/eval.rs`) execute a body as rustc's post-pass MIR says, instead of each
re-deriving drop flags, moves and unwinding on its own.

The code is `crates/hir-ty/src/mir/transform.rs` and `transform/`.

## Phases and queries

`MirBody::phase` is `MirPhase::Built` or `MirPhase::Runtime`.

- **Built** is what lowering produces, and what every IDE feature reads (`mir_body`,
  `monomorphized_mir_body`, borrowck, diagnostics). A `Drop` there drops what the place holds
  *if* it holds anything; no call or drop names where it unwinds to.
- **Runtime** is rustc's `MirPhase::Runtime(PostCleanup)` minus optimizations:
  - every `Drop` drops;
  - whether a drop runs is decided by the blocks around it, which read drop flags that are
    ordinary `bool` locals;
  - every call and drop outside a cleanup block names the cleanup block it unwinds to;
  - `MirBody::panic_cleanup` is where a statement that panics (index out of bounds, overflow,
    division by zero) unwinds to;
  - cleanup blocks end in `UnwindResume`.

The runtime bodies are separate salsa queries, so the IDE never computes them:
`runtime_mir_body`, `runtime_mir_body_for_closure`, `runtime_mir_body_for_coroutine`,
`runtime_mir_body_for_coroutine_by_move`, `runtime_promoted_mir_body` (the constant behind a
promoted `&expr`), and the `monomorphized_runtime_mir_body*` variants
(the passes run on the generic body; monomorphization substitutes after, as it does for built
bodies). Only codegen and the interpreter ask for them.

## What changed in lowering

**Move vs Copy** (`MirBody::mark_moves`, run at the end of every lowering). RA's lowering
wrote `Copy` for every operand that reads a place. Now an operand whose place type is not
`Copy` (modulo regions) is `Move`, as rustc's MIR building does. The one exception is the
operand of `[CONSTANT; N]`: lowering reads the constant's temporary with `Copy` because each
element is the constant made again, and `[CONSTANT; 0]` makes none. That `Copy` uses the
temporary up, and the passes treat it as a move that drops nothing.

**Constant indices from the end** count as rustc's do: `ConstantIndex { offset, from_end: true }`
with offset 1 is the last element. RA's lowering used 0.

## The passes, in order

1. **`cleanup::hoist_indexing`**: makes a terminator unwind only from what it calls. A
   terminator operand that reads through an `Index` projection, the destination of a call
   that is indexed, and an indexed drop get the indexing moved into a statement before them.
   That statement can panic, and it unwinds through `panic_cleanup`.
2. **`cleanup::add_cleanup_paths`**: builds the cleanup chain and points every call, drop and
   panicking statement outside cleanup at it.
   - The chain is one `Drop(local)` block per local the body drops, in reverse scope order:
     the order the code generator's landing pad used, ported from it.
   - A drop in a cleanup block has no unwind edge.
   - `place = value` on a field (a drop followed by the assignment) gets a cleanup block that
     still does the assignment before joining the chain, so that the place is not dropped
     twice: rustc's drop-and-replace.

   One chain for the whole body keeps the code generator's behaviour. rustc's per-scope drop
   trees give each call a shorter chain; the flags make the two equivalent in what they drop.
3. **`elaborate_drops::elaborate_drops`**: a port of rustc's `elaborate_drops` and
   `DropCtxt`.
   - *Move paths* (`move_paths.rs`) are created for:
     - every local that needs drop (promoted constants are never dropped);
     - every assigned or moved place inside it: fields of structs without a destructor, of
       tuples and closures; enum variants (`Downcast`); `Box` contents; constant indices and
       subslices of arrays, by element.
   - *Init analysis*: a forward maybe-init / maybe-uninit dataflow over the move paths.
     - Unwind edges are taken before each panicking statement and after each terminator's
       effects.
     - A call initializes its destination only on its normal edge.
     - Blocks in `unreached` are not executed.
   - *Each drop* is one of:
     - **dead** (`Goto`);
     - **static** (left as is);
     - **conditional** (a `SwitchInt` on the path's flag);
     - **open**: the value is taken apart:
       - struct, tuple and closure fields are dropped in order, each unwinding to the drops
         of the rest (rustc's drop ladder);
       - an enum is switched on its `Discriminant` and the variants are dropped field by
         field;
       - a `Box` drops its contents, then `<Box as Drop>::drop`, then its fields;
       - an array drops its elements.
   - *Flags*:
     - one `bool` per path that needs one, written right after each statement that
       initializes or moves out of it, so that a statement that panics unwinds with the state
       before it;
     - a call's destination flag is written on its normal edge;
     - flags are cleared after each elaborated drop, on the normal and the unwind edge;
     - flags are set at entry from the parameters.
4. **`packed_drops::add_moves_for_packed_drops`**: a drop of a field that a `#[repr(packed)]`
   struct leaves misaligned is turned into a move to an aligned temporary and a drop of that.

There are no optimization passes: no inlining, GVN, SROA, const propagation or simplification.

## What the consumers do with it

**ra-jit codegen** (`crates/hir-ty-codegen/mod.rs`) generates a Runtime body directly:
- no drop flags of its own;
- `Drop` is `drop_in_place` and a jump;
- each cleanup block a call unwinds to gets its own landing pad, which keeps the exception
  and jumps to the block;
- `UnwindResume` resumes with the kept exception.

The flag-following machinery that tried to decide drops statically was deleted
(`FlagEvent`, `Held`, `FlagFlow`, `follow_flags`, `forget_unread_flags`; 529 lines).

**The interpreter** runs Runtime bodies only, promoted constants included: a `Drop` drops. Its
place-set drop tracking (`DropFlags`) is deleted, and a body that is not elaborated (an `async`
body, which it can't run anyway) is reported as not supported.

## Not done yet

- **Coroutines.** `run_passes` returns `async` bodies unchanged. Codegen and layout still use
  the built body with their own drop flags, which live in the future's prefix and make some
  futures larger than rustc's. The next stage:
  - elaborate coroutine bodies, so their flags are ordinary locals and the held-locals
    analysis (`MirBody::locals_held_across_yields`) keeps them in the variants as rustc's
    `StateTransform` does;
  - give each `Yield` a drop path (rustc's coroutine drop tree), so that dropping a suspended
    future is MIR and not codegen's `drop_live_locals`;
  - lay the future out from the runtime body.

  After that the rest of codegen's legacy drop handling (`drop_flags`, `moved_paths`,
  deferred drops, `drop_live_locals`) can be deleted.
- **Abort on unwind.** A drop in a cleanup block that unwinds should abort (rustc's
  `UnwindAction::Terminate`); it continues unwinding today, which is what codegen did before.
  `nounwind` callees do not yet get `Terminate` edges.
- **Panics inside statements.** RA's MIR has no `Assert` terminator: `CheckedBinaryOp`
  panics on overflow and an `Index` projection panics out of bounds, in the middle of a
  statement. rustc's MIR computes the overflow bit or `Len` and `Lt`, then branches with
  `Assert`. Lowering both as rustc does would delete `panic_cleanup`, the
  `hoist_indexing` pass, and codegen's `index_spans` and `FakeRead` of an indexed place.
- **Subslices of arrays.** `Subslice { from, to }` always counts `to` from the end; rustc's
  has `from_end` and counts from the start for arrays.
- **Temporary drop order in lowering.** Temporaries are dropped where RA's lowering puts
  their scope ends. The differential tests and the fuzzer show no difference from rustc on
  the cases they cover, but no rustc-style scope tree was built.
