//! Native numerical core for aerospace-mcp.
//!
//! The MCP server, tool wrappers and data models stay in Python; this crate
//! implements the numerically heavy pieces that used to be pure Python:
//!
//! - [`rocket`]: 3-DOF rocket ascent simulation and performance analysis.
//! - [`optimize`]: golden-section search, projected gradient descent, and the
//!   launch-angle / thrust-profile optimizers that run the simulation in a
//!   loop without crossing back into Python per evaluation.
//! - [`orbit`]: Kepler's equation, element/state conversions, J2 propagation,
//!   ground tracks and Hohmann transfers.
//! - [`lambert`]: Izzo's (2015) Lambert solver, replacing the archived
//!   poliastro dependency.
//!
//! The PyO3 bindings live in a private `python` module behind the default
//! `python` feature and are exposed to Python as `aerospace_mcp._native`.
//!
//! WARNING: for educational and research purposes only. Do NOT use for real
//! flight planning, navigation, or spacecraft operations.

// Input checks are written as `!(x > 0.0)` on purpose so NaN is rejected.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod interp;
pub mod lambert;
pub mod optimize;
pub mod orbit;
pub mod rocket;
pub mod vec3;

#[cfg(feature = "python")]
mod python;
