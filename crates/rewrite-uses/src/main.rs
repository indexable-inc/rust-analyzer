//! Inline `use` imports as qualified paths and remove the `use` statements.
//!
//! Walks all local crates in a cargo workspace, and for each file:
//! 1. Collects all `use` items
//! 2. Resolves each imported name to its definition
//! 3. Finds all non-import references to that name
//! 4. Replaces each reference with the fully qualified path
//! 5. Removes the `use` statement

use hir::{Crate, FindPathConfig, Module, Semantics};
use ide_db::RootDatabase;
use syntax::{AstNode, TextRange, TextSize, ast};

fn main() -> anyhow::Result<()> {
    let root = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("no cwd"));

    let cargo_config = project_model::CargoConfig::default();
    let load_config = load_cargo::LoadCargoConfig {
        load_out_dirs_from_check: false,
        with_proc_macro_server: load_cargo::ProcMacroServerChoice::None,
        prefill_caches: false,
        num_worker_threads: 0,
        proc_macro_processes: 0,
    };

    eprintln!("loading workspace at {}", root.display());
    let (db, vfs, _proc_macro) = load_cargo::load_workspace_at(
        &root,
        &cargo_config,
        &load_config,
        &|msg| eprintln!("  {msg}"),
    )?;

    let sema = Semantics::new(&db);

    let mut total_inlined = 0usize;
    for krate in Crate::all(&db) {
        if !krate.origin(&db).is_local() {
            continue;
        }
        for module in krate.modules(&db) {
            let count = process_module(&db, &sema, &vfs, module)?;
            total_inlined += count;
        }
    }

    eprintln!("inlined {total_inlined} use imports");
    Ok(())
}

/// Process a single module: inline all use imports and write the file back.
/// Returns the number of use items inlined.
fn process_module(
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    vfs: &vfs::Vfs,
    module: Module,
) -> anyhow::Result<usize> {
    let Some(file_id) = module.as_source_file_id(db) else {
        return Ok(0);
    };

    let source_file = sema.parse(file_id);
    let use_items: Vec<ast::Use> = source_file
        .syntax()
        .children()
        .filter_map(ast::Use::cast)
        .collect();

    if use_items.is_empty() {
        return Ok(0);
    }

    // Collect all replacements: (range_to_replace -> replacement_text)
    let mut replacements: Vec<(TextRange, String)> = Vec::new();
    let mut use_ranges_to_remove: Vec<TextRange> = Vec::new();
    let mut inlined_count = 0;

    let find_path_cfg = FindPathConfig {
        prefer_no_std: false,
        prefer_prelude: false,
        prefer_absolute: false,
        allow_unstable: false,
    };

    let edition = module.krate(db).edition(db);

    for use_item in &use_items {
        let mut all_leaves_inlined = true;

        for_each_leaf_use_tree(use_item, &mut |use_tree| {
            let Some(path) = use_tree.path() else {
                all_leaves_inlined = false;
                return;
            };

            // Resolve the path to a definition
            let Some(resolution) = sema.resolve_path(&path) else {
                eprintln!(
                    "  warning: could not resolve path `{}`",
                    path.syntax().text()
                );
                all_leaves_inlined = false;
                return;
            };

            let def = match resolution {
                hir::PathResolution::Def(module_def) => {
                    ide_db::defs::Definition::from(module_def)
                }
                _ => {
                    all_leaves_inlined = false;
                    return;
                }
            };

            // Find the qualified path from this module
            let Some(item_in_ns) = def_to_item_in_ns(&def) else {
                all_leaves_inlined = false;
                return;
            };
            let Some(qual_path) = module.find_path(db, item_in_ns, find_path_cfg) else {
                eprintln!(
                    "  warning: could not find path for `{}`",
                    path.syntax().text()
                );
                all_leaves_inlined = false;
                return;
            };
            let qual_path_str = qual_path.display(db, edition).to_string();

            // Find all non-import usages of this definition in this file
            let usages = def
                .usages(sema)
                .in_scope(&ide_db::search::SearchScope::single_file(file_id))
                .all();

            let Some(file_refs) = usages.references.get(&file_id) else {
                // No usages found (the use itself was unused) — still remove it
                return;
            };

            for reference in file_refs {
                // Skip references that are inside use items (the import itself)
                if reference
                    .category
                    .contains(ide_db::search::ReferenceCategory::IMPORT)
                {
                    continue;
                }
                replacements.push((reference.range, qual_path_str.clone()));
            }
        });

        if all_leaves_inlined {
            // Remove the entire use item including trailing newline
            let range = use_item.syntax().text_range();
            let text = source_file.syntax().text().to_string();
            let end: usize = range.end().into();
            let remaining = &text[end..];
            let extra = remaining.len() - remaining.trim_start_matches([' ', '\n']).len();
            let extended =
                TextRange::new(range.start(), range.end() + TextSize::from(extra as u32));
            use_ranges_to_remove.push(extended);
            inlined_count += 1;
        }
    }

    if replacements.is_empty() && use_ranges_to_remove.is_empty() {
        return Ok(0);
    }

    // Merge use removals into replacements (use removals replace with "")
    for range in &use_ranges_to_remove {
        replacements.push((*range, String::new()));
    }

    // Sort by start position descending so we can apply in reverse without shifting
    replacements.sort_by(|a, b| b.0.start().cmp(&a.0.start()));

    // Apply edits to the file text (in reverse order to preserve positions)
    let original_text = source_file.syntax().text().to_string();
    let mut result = original_text.clone();

    for (range, replacement) in &replacements {
        let start: usize = range.start().into();
        let end: usize = range.end().into();
        result.replace_range(start..end, replacement);
    }

    if result != original_text {
        let vfs_file_id: vfs::FileId = file_id.file_id(db);
        let path = vfs.file_path(vfs_file_id);
        let path_str = path.as_path().expect("vfs path is not a real path");
        std::fs::write(path_str, &result)?;
        eprintln!("  wrote {path_str}");
    }

    Ok(inlined_count)
}

/// Iterate over all leaf `UseTree` nodes in a `Use` item.
fn for_each_leaf_use_tree(use_item: &ast::Use, f: &mut impl FnMut(&ast::UseTree)) {
    let Some(use_tree) = use_item.use_tree() else {
        return;
    };
    walk_use_tree(&use_tree, f);
}

fn walk_use_tree(tree: &ast::UseTree, f: &mut impl FnMut(&ast::UseTree)) {
    if let Some(list) = tree.use_tree_list() {
        for child in list.use_trees() {
            walk_use_tree(&child, f);
        }
    } else {
        f(tree);
    }
}

/// Convert a `Definition` to an `ItemInNs` for path finding.
fn def_to_item_in_ns(def: &ide_db::defs::Definition) -> Option<hir::ItemInNs> {
    Some(match def {
        ide_db::defs::Definition::Module(m) => hir::ItemInNs::Types((*m).into()),
        ide_db::defs::Definition::Function(f) => hir::ItemInNs::Values((*f).into()),
        ide_db::defs::Definition::Adt(a) => hir::ItemInNs::Types((*a).into()),
        ide_db::defs::Definition::Variant(v) => hir::ItemInNs::Values((*v).into()),
        ide_db::defs::Definition::Const(c) => hir::ItemInNs::Values((*c).into()),
        ide_db::defs::Definition::Static(s) => hir::ItemInNs::Values((*s).into()),
        ide_db::defs::Definition::Trait(t) => hir::ItemInNs::Types((*t).into()),
        ide_db::defs::Definition::TypeAlias(t) => hir::ItemInNs::Types((*t).into()),
        ide_db::defs::Definition::Macro(m) => hir::ItemInNs::Macros((*m).into()),
        _ => return None,
    })
}
