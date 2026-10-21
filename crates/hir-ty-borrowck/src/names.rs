//! How the messages name a place, as rustc's `describe_place`: `x`, `x.field`, `*x`, `x[_]`.

use hir_def::VariantId;
use rustc_type_ir::inherent::IntoKind;

use hir_ty::{
    mir::{LocalId, MirBody, PlaceElem, PlaceTy, ProjectionElem},
    next_solver::TyKind,
};

use super::Ctx;

impl<'db> Ctx<'_, 'db> {
    /// The name of `local` as the source has it: the variable, or the variable a closure
    /// captured.
    pub(super) fn local_name(&self, local: LocalId) -> Option<String> {
        let binding = match self.bindings.get(&local) {
            Some(&binding) => binding,
            None => self.upvars.iter().find(|it| it.local == local)?.binding,
        };
        Some(self.store[binding].name.as_str().to_owned())
    }

    /// The place as rustc writes it in a message, or `None` for a temporary.
    pub(super) fn describe(
        &self,
        body: &MirBody<'db>,
        local: LocalId,
        projection: &[PlaceElem],
    ) -> Option<String> {
        let mut start = 0;
        let mut name = match self.upvar_of_projection(body, local, projection) {
            Some((upvar, used)) => {
                start = used;
                self.store[self.upvars[upvar].binding].name.as_str().to_owned()
            }
            None => self.local_name(local)?,
        };
        let mut ty = self.place_ty(body, local, &projection[..start]);
        for (index, elem) in projection.iter().enumerate().skip(start) {
            match elem {
                ProjectionElem::Deref => {
                    let next_is_field =
                        matches!(projection.get(index + 1), Some(ProjectionElem::Field(_)));
                    if !next_is_field {
                        name = format!("*{name}");
                    }
                }
                ProjectionElem::Field(field) => {
                    name = format!("{name}.{}", self.field_name(ty, field.0));
                }
                ProjectionElem::Index(_) => name = format!("{name}[_]"),
                ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                    name = format!("{name}[..]")
                }
                ProjectionElem::Downcast(_) => (),
            }
            ty = self.project(ty, elem);
        }
        Some(name)
    }

    fn field_name(&self, ty: PlaceTy<'db>, field: u32) -> String {
        let variant = match (ty.variant_id, ty.ty.kind()) {
            (Some(variant), _) => variant,
            (None, TyKind::Adt(def, _)) if !def.is_enum() => {
                match VariantId::from_non_enum(def.def_id()) {
                    Some(it) => it,
                    None => return field.to_string(),
                }
            }
            _ => return field.to_string(),
        };
        let fields = variant.fields(self.db);
        match fields.fields().iter().nth(field as usize) {
            Some((_, data)) => data.name.as_str().to_owned(),
            None => field.to_string(),
        }
    }

    /// In a closure, the capture `local` with the first projections of a place stands for,
    /// and how many of the projections that is: `(*_1).0`, or the local lowering copied the
    /// capture into, a capture by reference behind the `Deref` after it.
    pub(super) fn upvar_of_projection(
        &self,
        body: &MirBody<'db>,
        local: LocalId,
        projection: &[PlaceElem],
    ) -> Option<(usize, usize)> {
        if let Some(upvar) = self.upvar_of_local(local) {
            let deref =
                self.upvars[upvar].by_ref && projection.first() == Some(&ProjectionElem::Deref);
            return Some((upvar, deref as usize));
        }
        if Some(local) != self.closure_local {
            return None;
        }
        let (field, used) = match projection {
            [ProjectionElem::Deref, ProjectionElem::Field(field), ..] => (field.0 as usize, 2),
            [ProjectionElem::Field(field), ..] => (field.0 as usize, 1),
            _ => return None,
        };
        let _ = body;
        let upvar = self.upvars.get(field)?;
        let deref = upvar.by_ref && projection.get(used) == Some(&ProjectionElem::Deref);
        Some((field, used + deref as usize))
    }
}
