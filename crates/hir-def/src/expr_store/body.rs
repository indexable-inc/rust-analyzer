//! Defines `Body`: a lowered representation of functions, statics and
//! consts.
use std::ops;

use base_db::SourceDatabase;
use hir_expand::{HirFileId, InFile, Lookup, span_map::SpanMap};
use span::{AstIdMap, Edition, ROOT_ERASED_FILE_AST_ID, RealSpanMap};
use syntax::{AstNode, GreenNode, NodeOrToken, SyntaxNode, SyntaxNodePtr, TextSize, ast};
use triomphe::Arc;

use crate::{
    DefWithBodyId, ExpressionStoreOwnerId, HasModule,
    expr_store::{
        ExpressionStore, ExpressionStoreSourceMap, Move, SelfParamPtr, lower::lower_body, pretty,
    },
    hir::{BindingId, ExprId, PatId},
    src::HasSource,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param<Id> {
    /// The id of the formal parameter. In analysis, you want to use this: it has a special status as a parameter,
    /// while [`user_written`][Self::user_written] is just a local variable.
    pub formal: Id,
    /// The id that corresponds to the pattern the user wrote. In IDE, you usually want to use this.
    ///
    /// It will be different than [`formal`][Self::formal] for coroutine fns (async fn etc.), because we apply
    /// a desugaring for those that copies the parameter into a local variable.
    pub user_written: Id,
}

impl<Id: Copy> Param<Id> {
    pub(crate) fn new(id: Id) -> Self {
        Self { formal: id, user_written: id }
    }
}

/// The body of an item (function, const etc.).
#[derive(Debug, Eq, PartialEq)]
pub struct Body {
    pub store: ExpressionStore,
    /// The patterns for the function's parameters. While the parameter types are
    /// part of the function signature, the patterns are not (they don't change
    /// the external type of the function).
    ///
    /// If this `Body` is for the body of a constant, this will just be
    /// empty.
    pub params: Box<[Param<PatId>]>,
    pub self_param: Option<Param<BindingId>>,
}

impl ops::Deref for Body {
    type Target = ExpressionStore;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.store
    }
}

/// An item body together with the mapping from syntax nodes to HIR expression
/// IDs. This is needed to go from e.g. a position in a file to the HIR
/// expression containing it; but for type inference etc., we want to operate on
/// a structure that is agnostic to the actual positions of expressions in the
/// file, so that we don't recompute types whenever some whitespace is typed.
///
/// One complication here is that, due to macro expansion, a single `Body` might
/// be spread across several files. So, for each ExprId and PatId, we record
/// both the HirFileId and the position inside the file. However, we only store
/// AST -> ExprId mapping for non-macro files, as it is not clear how to handle
/// this properly for macros.
#[derive(Default, Debug, Eq, PartialEq)]
pub struct BodySourceMap {
    pub self_param: Option<InFile<SelfParamPtr>>,
    pub store: ExpressionStoreSourceMap,
}

impl ops::Deref for BodySourceMap {
    type Target = ExpressionStoreSourceMap;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.store
    }
}

#[salsa::tracked]
impl Body {
    /// The body with where each part of it is written. For a body written in a file, this is
    /// [`Body::relative`] with its source map moved to where the body is in the file: the body
    /// is lowered once, and an edit elsewhere in the file only moves the map.
    #[salsa::tracked(lru = 512, returns(ref))]
    pub fn with_source_map(
        db: &dyn SourceDatabase,
        def: DefWithBodyId,
    ) -> (Arc<Body>, BodySourceMap) {
        let _p = tracing::info_span!("body_with_source_map_query").entered();
        let Some((body, source_map)) = Self::relative(db, def) else {
            let (body, source_map) = lower_in_file(db, def);
            return (Arc::new(body), source_map);
        };
        let syntax = body_syntax(db, def).as_ref().expect("lowered without its syntax");
        let item = def_syntax(db, def).value;
        let start = item.text_range().start() - syntax.item.text_range().start();
        let anchor = item
            .ancestors()
            .find(|it| it.text_range().start() == start && *it.green() == *syntax.green)
            .expect("the anchor is not around the item");
        debug_assert_eq!(Some(start), Self::anchor_offset(db, def));
        let to = |ptr: SyntaxNodePtr| {
            let range = ptr.text_range() + start;
            std::iter::successors(Some(anchor.clone()), |it| {
                it.child_or_token_at_range(range)?.into_node()
            })
            .find(|it| it.text_range() == range && it.kind() == ptr.kind())
            .map(|it| SyntaxNodePtr::new(&it))
            .expect("no node where the body has one")
        };
        let mv = Move { file: syntax.file_id, to: &to };
        let source_map = BodySourceMap {
            self_param: source_map.self_param.map(|it| mv.ptr(it)),
            store: source_map.store.moved(&mv),
        };
        if check_relative() {
            let (whole, in_file) = lower_in_file(db, def);
            assert_eq!(in_file, source_map, "{def:?}: the source map lowered alone, moved");
            assert_eq!(whole.params.len(), body.params.len());
        }
        (body.clone(), source_map)
    }

    /// The body lowered from [`body_syntax`] rather than from the whole file: what it is made of
    /// stays the same while edits elsewhere in the file move it, so it is lowered again only
    /// when its own text changes. The source map is of the tree of [`BodySyntax::root`]. `None`
    /// when the body is not written in a file, but by a macro. Kept for more bodies than
    /// [`Body::with_source_map`] is: what is lowered again after it is let go of has changed
    /// to all that reads it.
    #[salsa::tracked(lru = 4096, returns(ref))]
    pub fn relative(
        db: &dyn SourceDatabase,
        def: DefWithBodyId,
    ) -> Option<(Arc<Body>, BodySourceMap)> {
        let _p = tracing::info_span!("body_relative_query").entered();
        let syntax = body_syntax(db, def).as_ref()?;
        let node = syntax.item.to_node(&syntax.root());
        let (mut params, mut is_async_fn, mut is_gen_fn) = (None, false, false);
        let body = match def {
            DefWithBodyId::FunctionId(_) => {
                let it = ast::Fn::cast(node.clone())?;
                params = it.param_list();
                is_async_fn = it.async_token().is_some();
                is_gen_fn = it.gen_token().is_some();
                it.body().map(ast::Expr::from)
            }
            DefWithBodyId::ConstId(_) => ast::Const::cast(node.clone())?.body(),
            DefWithBodyId::StaticId(_) => ast::Static::cast(node.clone())?.body(),
            DefWithBodyId::VariantId(_) => ast::Variant::cast(node.clone())?.const_arg()?.expr(),
        };
        let maps = Some((&syntax.span_map, &syntax.ast_id_map));
        let (body, source_map) = lower_body(
            db,
            def,
            syntax.item,
            syntax.file_id,
            def.module(db),
            params,
            body,
            is_async_fn,
            is_gen_fn,
            maps,
        );
        Some((Arc::new(body), source_map))
    }

    #[salsa::tracked(returns(deref))]
    pub fn of(db: &dyn SourceDatabase, def: DefWithBodyId) -> Arc<Body> {
        match Self::relative(db, def) {
            Some((body, _)) => body.clone(),
            None => Self::with_source_map(db, def).0.clone(),
        }
    }
}

impl Body {
    /// Where in its file the anchor of [`body_syntax`] begins, which the source map of
    /// [`Body::relative`] is relative to. `None` for a body a macro writes.
    pub fn anchor_offset(db: &dyn SourceDatabase, def: DefWithBodyId) -> Option<TextSize> {
        let syntax = body_syntax(db, def).as_ref()?;
        Some(def_syntax(db, def).value.text_range().start() - syntax.item.text_range().start())
    }

    /// The source map of [`Body::relative`] when there is one, and of
    /// [`Body::with_source_map`] when there is not: for what needs of it only the file or the
    /// expansion each part is written in, and the macro calls, and not where in the file.
    /// It stays the same while edits elsewhere in the file move the body.
    pub fn unmoved_source_map(db: &dyn SourceDatabase, def: DefWithBodyId) -> &BodySourceMap {
        match Self::relative(db, def) {
            Some((_, source_map)) => source_map,
            None => &Self::with_source_map(db, def).1,
        }
    }
}

/// The syntax of the item `def` is the body of.
fn def_syntax(db: &dyn SourceDatabase, def: DefWithBodyId) -> InFile<SyntaxNode> {
    match def {
        DefWithBodyId::FunctionId(it) => it.lookup(db).source(db).map(|it| it.syntax().clone()),
        DefWithBodyId::ConstId(it) => it.lookup(db).source(db).map(|it| it.syntax().clone()),
        DefWithBodyId::StaticId(it) => it.lookup(db).source(db).map(|it| it.syntax().clone()),
        DefWithBodyId::VariantId(it) => it.lookup(db).source(db).map(|it| it.syntax().clone()),
    }
}

/// Lowers the body of `def` from the whole file it is in. What a body written by a macro is
/// lowered with, and what [`check_relative`] checks the others against.
fn lower_in_file(db: &dyn SourceDatabase, def: DefWithBodyId) -> (Body, BodySourceMap) {
    let mut params = None;
    let mut is_async_fn = false;
    let mut is_gen_fn = false;
    let (InFile { file_id, value: body }, syntax_node) = match def {
        DefWithBodyId::FunctionId(f) => {
            let src = f.lookup(db).source(db);
            params = src.value.param_list();
            is_async_fn = src.value.async_token().is_some();
            is_gen_fn = src.value.gen_token().is_some();
            let syntax_node = SyntaxNodePtr::new(src.syntax().value);
            (src.map(|it| it.body().map(ast::Expr::from)), syntax_node)
        }
        DefWithBodyId::ConstId(c) => {
            let src = c.lookup(db).source(db);
            let syntax_node = SyntaxNodePtr::new(src.syntax().value);
            (src.map(|it| it.body()), syntax_node)
        }
        DefWithBodyId::StaticId(s) => {
            let src = s.lookup(db).source(db);
            let syntax_node = SyntaxNodePtr::new(src.syntax().value);
            (src.map(|it| it.body()), syntax_node)
        }
        DefWithBodyId::VariantId(v) => {
            let src = v.lookup(db).source(db);
            let syntax_node = SyntaxNodePtr::new(src.syntax().value);
            (src.map(|it| it.const_arg()?.expr()), syntax_node)
        }
    };
    let module = def.module(db);
    lower_body(db, def, syntax_node, file_id, module, params, body, is_async_fn, is_gen_fn, None)
}

/// Whether [`Body::with_source_map`] checks the source map of each body lowered alone, moved,
/// against that of the same body lowered in its file (`RA_CHECK_RELATIVE_BODIES`).
fn check_relative() -> bool {
    static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CHECK.get_or_init(|| std::env::var_os("RA_CHECK_RELATIVE_BODIES").is_some())
}

/// What lowering a body reads of the file it is written in, as a tree of its own: the anchor
/// the body is written in ([`RealSpanMap::anchor_at`]), which is the item itself unless that
/// is an item in a body, or the item of an `impl` or a module with attributes. Its ranges begin
/// at the anchor, and so stay as they are while edits elsewhere move it in the file. A body
/// lowered from this is lowered again only when this changes, not on every edit of the file.
#[derive(Debug, PartialEq, Eq)]
pub struct BodySyntax {
    pub file_id: HirFileId,
    green: GreenNode,
    /// The item in [`BodySyntax::root`].
    pub item: SyntaxNodePtr,
    pub ast_id_map: AstIdMap,
    pub span_map: RealSpanMap,
}

impl BodySyntax {
    /// The anchor as the root of a tree of its own.
    pub fn root(&self) -> SyntaxNode {
        SyntaxNode::new_root(self.green.clone())
    }
}

/// The [`BodySyntax`] of a body written in a file, `None` for one a macro writes.
#[salsa::tracked(returns(ref))]
pub fn body_syntax(db: &dyn SourceDatabase, def: DefWithBodyId) -> Option<BodySyntax> {
    let src = def_syntax(db, def);
    let SpanMap::RealSpanMap(span_map) = src.file_id.span_map(db) else { return None };
    let item = src.value;
    let (start, anchor) = span_map.anchor_at(item.text_range().start());
    let anchor = match anchor == ROOT_ERASED_FILE_AST_ID {
        true => item.ancestors().last()?,
        false => {
            let ptr = src.file_id.ast_id_map(db).get_erased(anchor);
            item.ancestors().find(|it| SyntaxNodePtr::new(it) == ptr)?
        }
    };
    let root = SyntaxNode::new_root(anchor.green().to_owned());
    let range = item.text_range() - start;
    let relative = match root.covering_element(range) {
        NodeOrToken::Node(it) => it,
        NodeOrToken::Token(it) => it.parent()?,
    };
    let relative =
        relative.ancestors().find(|it| it.text_range() == range && it.kind() == item.kind())?;
    Some(BodySyntax {
        file_id: src.file_id,
        green: anchor.green().to_owned(),
        item: SyntaxNodePtr::new(&relative),
        ast_id_map: src.file_id.ast_id_map(db).relative_to(&anchor, &root),
        span_map: span_map.within(anchor.text_range()),
    })
}

impl Body {
    pub fn root_expr(&self) -> ExprId {
        // A `Body` can also contain root expressions that aren't the body (in the param patterns),
        // but the body always come last.
        self.store.expr_roots().next_back().unwrap()
    }

    /// Returns `true` if this is the formal or user-written self param.
    #[inline]
    pub fn is_any_self_param(&self, binding: BindingId) -> bool {
        self.self_param.is_some_and(|self_param| {
            self_param.formal == binding || self_param.user_written == binding
        })
    }

    pub fn pretty_print(
        &self,
        db: &dyn SourceDatabase,
        owner: DefWithBodyId,
        edition: Edition,
    ) -> String {
        pretty::print_body_hir(db, self, owner, edition)
    }

    pub fn pretty_print_expr(
        &self,
        db: &dyn SourceDatabase,
        owner: DefWithBodyId,
        expr: ExprId,
        edition: Edition,
    ) -> String {
        pretty::print_expr_hir(db, self, owner.into(), expr, edition)
    }

    pub fn pretty_print_pat(
        &self,
        db: &dyn SourceDatabase,
        owner: ExpressionStoreOwnerId,
        pat: PatId,
        oneline: bool,
        edition: Edition,
    ) -> String {
        pretty::print_pat_hir(db, self, owner, pat, oneline, edition)
    }
}

impl BodySourceMap {
    pub fn self_param_syntax(&self) -> Option<InFile<SelfParamPtr>> {
        self.self_param
    }
}
