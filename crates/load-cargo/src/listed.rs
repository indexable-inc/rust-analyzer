//! Proc-macro libraries that list their macros in a static, as ra-jit links them (`--shared`),
//! which the toolchain's server does not read: a caller that names a server reading that list
//! has such libraries expanded by it, and the others by the toolchain's.

use std::sync::OnceLock;

use vfs::{AbsPath, AbsPathBuf};

/// The name of the static the list is in.
const LIST: &[u8] = b"RA_JIT_PROC_MACROS";

static SERVER: OnceLock<AbsPathBuf> = OnceLock::new();

/// For the sysroot of a workspace and libraries rustc compiled, the library that lists its
/// macros a caller has of each, where it has one.
pub type Own = dyn Fn(Option<&AbsPath>, &[AbsPathBuf]) -> Vec<Option<AbsPathBuf>> + Send + Sync;

static OWN: OnceLock<Box<Own>> = OnceLock::new();

/// Has each library a load names that does not list its macros expanded with the one `own`
/// has of it, by the server [`serve_listed_with`] named. The first call is the one kept.
pub fn own_libraries_with(own: Box<Own>) {
    _ = OWN.set(own);
}

/// The library listing its macros that stands for each of `libraries`, where there is one.
pub(crate) fn own(sysroot: Option<&AbsPath>, libraries: &[AbsPathBuf]) -> Vec<Option<AbsPathBuf>> {
    match (OWN.get(), server()) {
        (Some(own), Some(_)) if !libraries.is_empty() => own(sysroot, libraries),
        _ => vec![None; libraries.len()],
    }
}

/// Has the libraries with a list of their macros loaded from here on expanded by the server
/// at `server`. The first call is the one kept.
pub fn serve_listed_with(server: AbsPathBuf) {
    _ = SERVER.set(server);
}

pub(crate) fn server() -> Option<&'static AbsPath> {
    SERVER.get().map(AbsPathBuf::as_path)
}

/// Whether the library at `path` lists its macros: whether the name of the list is in it,
/// which a library rustc built has only when its own source names it.
pub fn is_listed(path: &AbsPath) -> bool {
    std::fs::read(path).is_ok_and(|bytes| bytes.windows(LIST.len()).any(|it| it == LIST))
}
