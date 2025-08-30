# Product Requirements: Advanced Aerospace MCP Tools

## Overview
Expand this MCP server beyond flight planning to expose research-grade tools across aircraft, drones, rockets (guided/unguided), and spacecraft. Favor lightweight, reproducible calculations with optional heavy integrations. All tools return structured, unit-consistent results and graceful fallbacks when an optional dependency is missing.

## Goals
- Provide domain-focused MCP tools with clear inputs/outputs and SI units.
- Wrap widely used Python libraries to accelerate aerospace research workflows.
- Keep core server responsive; long jobs run with timeouts and clear errors.

## Non-Goals
- Real-time flight operations, ATC integration, or safety-critical usage.
- Full CFD/FEA pipelines; provide lower-order aero methods and hooks instead.

## Tool Catalog (Proposed)
Notes: Tools are grouped; each lists candidate libraries. Prefer Tier 1 (light) first.

1) Common Atmosphere & Frames
- get_atmosphere_profile: ISA/COESA properties vs altitude, lapse-rate banding.
  Libraries: ambiance (Tier 1), aerosandbox.atmosphere (Tier 1)
- wind_model_simple: logarithmic or power-law profile for low-altitude studies.
  Libraries: none (in-house), optional: windscrape+xarray (Tier 3)
- transform_frames: ECI/ECEF/ITRF conversions, timescales.
  Libraries: astropy (Tier 1), skyfield (Tier 1), spiceypy (Tier 2)

2) Aircraft Aerodynamics & Propulsion
- wing_vlm: VLM-based CL/CD/CM, stability derivatives.
  Libraries: AeroSandbox (Tier 1), MachUpX (Tier 1)
- airfoil_polar: Airfoil polars via XFoil wrapper or surrogate.
  Libraries: aerosandbox (built-in XFoil), xfoil-python (Tier 2)
- turbofan_cycle: On-design/off-design thrust and SFC maps.
  Libraries: pyCycle + OpenMDAO (Tier 2)
- mission_performance: Mission fuel/time using Breguet + OpenAP.
  Libraries: OpenAP (existing), aerosandbox (mission models)

3) Drone/Propeller Systems
- propeller_bemt: Thrust/torque/eta vs RPM, diameter, pitch.
  Libraries: AeroSandbox (BEMT), pybemt (Tier 2)
- uav_energy_estimate: Multirotor endurance/range vs mass, battery, wind.
  Libraries: in-house + aerosandbox utilities
- motor_match: Motor/ESC/prop selection sweep.
  Libraries: in-house; data-driven configs

4) Rockets (Unguided/Guided)
- rocket_mass_budget: Stage mass fractions and ∆v (Tsiolkovsky).
  Libraries: in-house; RocketCEA for Isp (Tier 2)
- rocket_vertical_ascent_3dof: Ballistic/lofted ascent with drag and winds.
  Libraries: RocketPy (Tier 1)
- launch_trajectory_opt: Pitch program optimization with path constraints.
  Libraries: Dymos + OpenMDAO (Tier 2), casadi (Tier 2)

5) Spacecraft & Orbital Design
- lambert_solve: Two-body Lambert solutions and transfer elements.
  Libraries: poliastro (Tier 1), pykep (Tier 2)
- hohmann_bielliptic: ∆v budgets and timing.
  Libraries: poliastro (Tier 1)
- orbit_propagate: Keplerian + J2 propagation, ephemerides.
  Libraries: poliastro+astropy (Tier 1), orekit (Tier 2), spiceypy (Tier 2)
- porkchop_grid: Departure/arrival grid with C3/∆v contours.
  Libraries: poliastro (Tier 1)

6) GNC & Controls
- kalman_filter_sim: IMU/GNSS fusion (EKF/UKF) on sample trajectory.
  Libraries: filterpy (Tier 1)
- control_design: LQR/PID synthesis for linearized models.
  Libraries: python-control (Tier 1), slycot optional

## API Shape (MCP)
- Each tool: name, JSON schema input, deterministic text/JSON output.
- Units: SI by default; accept/return unit tags when practical.
- Capability discovery: get_system_status lists available domains and missing deps.
- Timeouts: default 10–60s; return partials or errors with actionable hints.

## Library Integration Plan
- Add optional extras to `pyproject.toml`:
  - aircraft: aerosandbox, machupx
  - rockets: rocketpy, rocketcea
  - space: poliastro, astropy, spiceypy
  - trajopt: openmdao, dymos, casadi
  - gnc: filterpy, control
  - frames: skyfield
- Lazy import inside wrappers; set AVAILABLE flags per feature.
- For heavy stacks (OpenMDAO/Dymos, Orekit), mark Tier 2 and degrade gracefully.

## Implementation Plan (for AI Coding Tool)
1) Scaffolding
- Create `aerospace_mcp/integrations/` with modules:
  - atmosphere.py, frames.py, aero.py, propellers.py, rockets.py, orbits.py, gnc.py, trajopt.py
- Add `__all__` and small pydantic models for IO data structures.

2) Optional Dependencies
- Update `pyproject.toml` with `[project.optional-dependencies]` groups listed above.
- Add import probes: e.g., `AEROSANDBOX_AVAILABLE`, `POLIASTRO_AVAILABLE`.
- Extend `get_system_status` to report availability counts/versions.

3) Wrappers (Tier 1 first)
- atmosphere.get_atmosphere_profile(altitudes_m: list[float]) -> list[dict]
- frames.transform_frames(epoch_iso: str, xyz: list[float], from: str, to: str) -> dict
- aero.wing_vlm(planform: dict, alpha_deg: list[float], mach: float) -> dict
- propellers.propeller_bemt(params: dict, rpm: list[float]) -> dict
- rockets.rocket_mass_budget(stages: list[dict]) -> dict
- rockets.vertical_ascent_3dof(vehicle: dict, env: dict) -> dict  (RocketPy)
- orbits.lambert_solve(r1_km: list[float], r2_km: list[float], tof_s: float, mu_km3s2: float) -> dict
- orbits.hohmann_bielliptic(elements: dict) -> dict
- orbits.orbit_propagate(elements: dict, tspan_s: float) -> list[dict]
- gnc.kalman_filter_sim(traj_true: list[dict], sensors: dict) -> dict

4) MCP Tool Definitions
- Add new Tool entries in `aerospace_mcp/server.py` with JSON Schemas mirroring the wrappers.
- Standardize responses: headline, key scalars, and optional tabular blocks.
- Respect timeouts with `asyncio.wait_for`; catch ImportError and return guidance.

5) Tests
- Add unit tests per wrapper using synthetic inputs and golden values (small tolerances).
- Skip tests when dependency unavailable via `pytest.importorskip`.
- Add integration tests for representative tools (e.g., lambert_solve, propeller_bemt).

6) Docs & Examples
- Update README “Features” to include new domains.
- Add docs pages under `docs/` with usage examples and input templates.
- Provide quick recipes (e.g., LEO transfer, 3DOF rocket ascent, UAV endurance).

7) Quality & Safety
- Enforce SI units and document conversions; include NM/KM helpers already present.
- Input validation via pydantic; clamp unphysical values with warnings.
- Deterministic seeds where randomized.
- Performance: add `functools.lru_cache` for atmospheric/ephemeris helpers.

## Acceptance Criteria
- Tier 1 tools implemented and discoverable via `get_system_status`.
- Meaningful outputs for at least:
  - Atmosphere profile
  - Wing VLM sweep or Airfoil polar
  - Propeller BEMT sweep
  - Rocket mass budget
  - 3DOF rocket ascent (basic)
  - Lambert transfer + Hohmann
  - Simple EKF demo
- Tests pass in a “core” environment (no Tier 2 deps) with coverage on wrappers.

## Risks & Mitigations
- Heavy deps (OpenMDAO/Orekit) complicate installs → keep optional, provide stubs.
- Numerical instability (optimizers) → constrain defaults, document solver options.
- Ephemeris data requirements → clearly document SPICE kernels; default to simplified models.

## Milestones
- M1 (Core, 1–2 weeks): Atmosphere, Lambert/Hohmann, Rocket budget, Propeller BEMT.
- M2 (Aircraft/Drone, 2 weeks): VLM wing, UAV energy, EKF.
- M3 (Rockets/TrajOpt, 3–4 weeks): 3DOF ascent, basic pitch optimization.
- M4 (Space/Advanced, 3–4 weeks): Porkchop, J2 propagation, SPICE/OREKIT optional.

