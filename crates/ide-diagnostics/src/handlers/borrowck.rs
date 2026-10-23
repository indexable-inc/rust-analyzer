use syntax::{TextRange, TextSize};

use crate::{Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: borrowck
//
// An error of rustc's borrow checker: a use of a moved or uninitialized value, a mutation of
// what is not mutable, a borrow that conflicts with another or does not live long enough.
pub(crate) fn borrowck_error(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::BorrowckError,
) -> Diagnostic {
    // rustc gives some errors no code, as "lifetime may not live long enough".
    let code = match d.code {
        "" => DiagnosticCode::Ra("borrowck", crate::Severity::Error),
        code => DiagnosticCode::RustcHardError(code),
    };
    let mut diagnostic = Diagnostic::new_with_syntax_node_ptr(ctx, code, d.message.clone(), d.node);
    // rustc's span of an expression a macro writes, `&$e`, is in the macro's definition, where
    // its first token is; one the caller writes is the caller's. A definition in another file
    // stays at the call.
    if let Some(file) = d.node.file_id.macro_file() {
        let start = d.node.value.text_range().start();
        let (origin, context) = hir::InMacroFile::new(file, start).original_file_range(ctx.sema.db);
        if !context.is_root() && origin.file_id.file_id(ctx.sema.db) == diagnostic.range.file_id {
            diagnostic.range.range = origin.range;
        }
    }
    if d.at_end {
        let range = diagnostic.range.range;
        if let Some(last) = range.end().checked_sub(TextSize::new(1))
            && last >= range.start()
        {
            diagnostic.range.range = TextRange::new(last, range.end());
        }
    }
    diagnostic.stable()
}

// Diagnostic: borrowck-skipped
//
// The borrow checker did not check a body, because MIR lowering does not support something
// in it. Reported only for the borrow checker's test harness.
pub(crate) fn borrowck_skipped(
    ctx: &DiagnosticsContext<'_, '_>,
    d: &hir::BorrowckSkipped,
) -> Diagnostic {
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        DiagnosticCode::Ra("borrowck-skipped", crate::Severity::WeakWarning),
        format!("not borrow checked: {}", d.reason),
        d.node,
    )
}
