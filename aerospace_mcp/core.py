"""Core business logic layer for the aerospace flight planning system.

This module provides the shared domain logic consumed by both the HTTP (FastAPI)
and MCP (Model Context Protocol) interface layers.  It includes:

- Pydantic data models for airports, flight plan requests, and responses.
- Airport resolution from IATA codes or city-name searches.
- Great-circle (geodesic) route generation on the WGS-84 ellipsoid.
- Flight performance estimation via the OpenAP aircraft performance library,
  including climb / cruise / descent segment modelling and fuel-burn calculations.

WARNING: This module is for educational and research purposes only.
Do NOT use for real flight planning, navigation, or aircraft operations.
All estimates assume zero-wind conditions and simplified performance models
that are not certified for operational use.
"""

from __future__ import annotations

import math
from typing import Literal

import airportsdata
from geographiclib.geodesic import Geodesic
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Optional / graceful import for OpenAP (perf + fuel)
# ---------------------------------------------------------------------------
# OpenAP may not be installed in every deployment (e.g. lightweight containers
# or serverless runtimes).  We attempt the import and set a module-level flag
# so that callers can check availability before invoking performance functions.
OPENAP_AVAILABLE = True
try:
    from openap import FuelFlow, prop
    from openap.gen import FlightGenerator
except Exception:
    OPENAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Unit-conversion constants
# ---------------------------------------------------------------------------
# 1 nautical mile (NM) = 1.852 km exactly, by international definition.
# NM_PER_KM: multiply a distance in kilometres to get nautical miles.
NM_PER_KM = 0.539956803  # 1 / 1.852
# KM_PER_NM: multiply a distance in nautical miles to get kilometres.
KM_PER_NM = 1.0 / NM_PER_KM  # ~1.852


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class AirportOut(BaseModel):
    """Output model representing an airport with geographic and identification data.

    Contains the IATA/ICAO codes, human-readable name and city, the ISO
    country code, WGS-84 latitude/longitude, and an optional IANA timezone
    string.  Instances are typically produced by the airport-resolution
    helpers and returned to callers via the API or MCP interfaces.
    """

    iata: str
    icao: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    tz: str | None = None


class PlanRequest(BaseModel):
    """Input model for flight planning requests with aircraft and route configuration.

    Callers specify departure and arrival locations by city name (with an
    optional ISO alpha-2 country code for disambiguation) or by providing an
    explicit IATA code via the ``prefer_depart_iata`` / ``prefer_arrive_iata``
    fields.  Aircraft type, cruise altitude, mass override, and polyline
    sampling resolution can all be configured.

    Attributes:
        depart_city: Departure city name (e.g. ``"San Jose"``).
        arrive_city: Arrival city name (e.g. ``"Tokyo"``).
        depart_country: Optional ISO alpha-2 country code to disambiguate departure.
        arrive_country: Optional ISO alpha-2 country code to disambiguate arrival.
        prefer_depart_iata: If set, forces a specific departure airport by IATA code.
        prefer_arrive_iata: If set, forces a specific arrival airport by IATA code.
        ac_type: ICAO aircraft type designator (e.g. ``"A320"``, ``"B738"``).
        cruise_alt_ft: Cruise altitude in feet (default 35 000 ft).
        mass_kg: Aircraft mass in kg; defaults to 85 % MTOW when ``None``.
        route_step_km: Sampling interval in km for the great-circle polyline.
        backend: Performance-estimation backend (currently only ``"openap"``).
    """

    # You can pass cities, or override with explicit IATA
    depart_city: str = Field(..., description="e.g., 'San Jose'")
    arrive_city: str = Field(..., description="e.g., 'Tokyo'")
    depart_country: str | None = Field(
        None, description="ISO alpha-2 country code (optional)"
    )
    arrive_country: str | None = Field(
        None, description="ISO alpha-2 country code (optional)"
    )
    prefer_depart_iata: str | None = Field(
        None, description="Force a particular departure airport by IATA"
    )
    prefer_arrive_iata: str | None = Field(
        None, description="Force a particular arrival airport by IATA"
    )

    # Aircraft/performance knobs
    ac_type: str = Field(..., description="ICAO aircraft type (e.g., 'A320', 'B738')")
    cruise_alt_ft: int = Field(35000, ge=8000, le=45000)
    mass_kg: float | None = Field(
        None, description="If not set, defaults to 85% MTOW when available"
    )
    route_step_km: float = Field(
        25.0, gt=1.0, description="Sampling step for polyline points"
    )
    backend: Literal["openap"] = "openap"  # Placeholder for future backends


class SegmentEst(BaseModel):
    """Performance estimates for a single flight segment (climb, cruise, or descent).

    Each segment captures the elapsed time, ground distance covered, average
    ground speed, and total fuel consumed.  Values are computed under
    zero-wind assumptions using the OpenAP performance library.

    Attributes:
        time_min: Duration of the segment in minutes.
        distance_km: Ground distance covered during the segment in kilometres.
        avg_gs_kts: Average ground speed in knots (nautical miles per hour).
        fuel_kg: Estimated fuel burn for the segment in kilograms.
    """

    time_min: float
    distance_km: float
    avg_gs_kts: float
    fuel_kg: float


class PlanResponse(BaseModel):
    """Complete flight plan response with route, airports, and performance estimates.

    Returned by the flight-planning functions after successfully resolving
    airports, computing the great-circle route, and generating performance
    estimates.

    Attributes:
        engine: Name of the performance backend used (e.g. ``"openap"``).
        depart: Resolved departure airport details.
        arrive: Resolved arrival airport details.
        distance_km: Total great-circle distance in kilometres.
        distance_nm: Total great-circle distance in nautical miles.
        polyline: Ordered list of ``(lat, lon)`` waypoints along the route.
        estimates: Dictionary containing block totals and per-segment breakdowns
            (``"block"``, ``"climb"``, ``"cruise"``, ``"descent"``, ``"assumptions"``).
    """

    engine: str
    depart: AirportOut
    arrive: AirportOut
    distance_km: float
    distance_nm: float
    polyline: list[tuple[float, float]]  # [(lat, lon), ...]
    estimates: dict  # {"block": {...}, "climb": {...}, ...}


# ---------------------------------------------------------------------------
# Airport data loading
# ---------------------------------------------------------------------------
# Load the full IATA-keyed airport dictionary once at module import time.
# This is a fast, in-process operation with no network calls -- the data
# ships bundled inside the ``airportsdata`` package.
# Shape example:
#   {'SJC': {'iata':'SJC', 'icao':'KSJC', 'name':'San Jose Intl',
#            'city':'San Jose', 'country':'US', 'lat':..., 'lon':...}, ...}
_AIRPORTS_IATA = airportsdata.load("IATA")


# ---------------------------------------------------------------------------
# Airport resolution functions
# ---------------------------------------------------------------------------


def _airport_from_iata(iata: str) -> AirportOut | None:
    """Look up a single airport by its IATA code from the in-memory database.

    The lookup is case-insensitive; the code is upper-cased before querying.

    Args:
        iata: Three-letter IATA airport code (e.g. ``"SJC"``, ``"NRT"``).

    Returns:
        An ``AirportOut`` instance if the code exists in the database,
        or ``None`` if no match is found.
    """
    ap = _AIRPORTS_IATA.get(iata.upper())
    if not ap:
        return None
    return AirportOut(
        iata=iata.upper(),
        icao=ap.get("icao", ""),
        name=ap.get("name", ""),
        city=ap.get("city", ""),
        country=ap.get("country", ""),
        lat=float(ap["lat"]),
        lon=float(ap["lon"]),
        tz=ap.get("tz"),
    )


def _find_city_airports(city: str, country: str | None = None) -> list[AirportOut]:
    """Search airports by city name with optional country filter.

    Performs a case-insensitive scan of the in-memory IATA database.  An
    airport matches if its ``city`` field equals the query exactly, **or**
    the query appears as a substring of the airport ``name`` (useful for
    names like "San Jose International").

    Results are sorted with a preference heuristic: airports whose name
    contains the word "International" are ranked first, followed by
    alphabetical ordering.  This helps callers pick the most likely
    commercial airport when multiple matches exist for a city.

    Args:
        city: City name to search for (e.g. ``"Tokyo"``).
        country: Optional ISO alpha-2 country code (e.g. ``"JP"``) to narrow
            results.  When ``None``, all countries are searched.

    Returns:
        A list of ``AirportOut`` instances matching the query, sorted by
        international-preference heuristic.  May be empty if no matches
        are found.
    """
    city_l = city.strip().lower()
    if not city_l:  # Return empty list for empty city names
        return []
    out = []
    for iata, ap in _AIRPORTS_IATA.items():
        # Skip entries without a valid IATA code (heliports, closed, etc.)
        if not iata or not ap.get("iata"):
            continue
        # Match on exact city name or substring of airport name
        if (
            ap.get("city", "").strip().lower() == city_l
            or city_l in ap.get("name", "").lower()
        ):
            # Optionally restrict to a specific country
            if country is None or (ap.get("country", "").upper() == country.upper()):
                out.append(
                    AirportOut(
                        iata=iata.upper(),
                        icao=ap.get("icao", ""),
                        name=ap.get("name", ""),
                        city=ap.get("city", ""),
                        country=ap.get("country", ""),
                        lat=float(ap["lat"]),
                        lon=float(ap["lon"]),
                        tz=ap.get("tz"),
                    )
                )
    # Heuristic sort: airports with "International" in the name float to the
    # top (False < True, so we negate the test).  Ties broken alphabetically.
    out.sort(key=lambda a: ("international" not in a.name.lower(), a.name))
    return out


class AirportResolutionError(Exception):
    """Raised when an airport cannot be resolved from the given input.

    This may happen if an explicit IATA code is invalid or if no airports
    match a city/country query.
    """

    pass


def _resolve_endpoint(
    city: str,
    country: str | None,
    prefer_iata: str | None,
    role: str,
) -> AirportOut:
    """Resolve a flight endpoint from either an explicit IATA code or city/country search.

    If ``prefer_iata`` is provided the lookup is performed directly against
    the IATA database.  Otherwise, a city-based search is executed and the
    top-ranked result (international airports preferred) is returned.

    Args:
        city: City name for the endpoint (used only when ``prefer_iata`` is
            ``None``).
        country: Optional ISO alpha-2 country code for disambiguation.
        prefer_iata: If set, overrides the city search and resolves this
            IATA code directly.
        role: Human-readable label (``"departure"`` or ``"arrival"``) used in
            error messages to help callers identify which endpoint failed.

    Returns:
        The resolved ``AirportOut`` for the endpoint.

    Raises:
        AirportResolutionError: If the IATA code is not found, or if no
            airports match the city/country query.
    """
    if prefer_iata:
        ap = _airport_from_iata(prefer_iata)
        if not ap:
            raise AirportResolutionError(f"{role}: IATA '{prefer_iata}' not found.")
        return ap

    cands = _find_city_airports(city, country)
    if not cands:
        raise AirportResolutionError(
            f"{role}: no airport for city='{city}' (country={country or 'ANY'})."
        )
    # Return the highest-ranked candidate (international airports first).
    return cands[0]


# ---------------------------------------------------------------------------
# Geodesic / polyline generation
# ---------------------------------------------------------------------------


def great_circle_points(
    lat1: float, lon1: float, lat2: float, lon2: float, step_km: float
) -> tuple[list[tuple[float, float]], float]:
    """Generate evenly-spaced points along the great-circle path between two coordinates.

    Uses the WGS-84 ellipsoid model via ``geographiclib`` to compute the
    geodesic (shortest-path) line between the two endpoints, then samples
    ``(lat, lon)`` positions at approximately ``step_km``-kilometre intervals.

    Args:
        lat1: Latitude of the starting point in decimal degrees.
        lon1: Longitude of the starting point in decimal degrees.
        lat2: Latitude of the ending point in decimal degrees.
        lon2: Longitude of the ending point in decimal degrees.
        step_km: Desired spacing between sampled points in kilometres.
            The actual spacing is adjusted so that the arc is divided into
            an integer number of equal-length segments.

    Returns:
        A two-element tuple:
            - A list of ``(lat, lon)`` tuples representing the sampled polyline,
              including both endpoints.
            - The total geodesic distance in kilometres.
    """
    # Solve the inverse geodesic problem to get total distance and initial azimuth.
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    dist_m = g["s12"]  # total geodesic distance in metres

    # Build a geodesic line object from the start point along the initial azimuth.
    line = Geodesic.WGS84.Line(lat1, lon1, g["azi1"])

    # Determine the number of segments: divide total distance by the requested
    # step size, rounding up so that each segment is <= step_km.
    n = max(1, int(math.ceil((dist_m / 1000.0) / step_km)))

    # Sample n+1 points (including both endpoints) at equal arc-length intervals.
    pts = []
    for i in range(n + 1):
        # Distance along the arc for this sample; clamped to dist_m to avoid
        # floating-point overshoot past the destination.
        s = min(dist_m, (dist_m * i) / n)
        p = line.Position(s)
        pts.append((p["lat2"], p["lon2"]))
    return pts, dist_m / 1000.0


# ---------------------------------------------------------------------------
# OpenAP estimates (climb / cruise / descent)
# ---------------------------------------------------------------------------


class OpenAPError(Exception):
    """Raised when an OpenAP operation fails.

    Common causes include the OpenAP package not being installed, an
    unrecognised aircraft type, or an internal modelling error within the
    OpenAP library.
    """

    pass


def estimates_openap(
    ac_type: str, cruise_alt_ft: int, mass_kg: float | None, route_dist_km: float
) -> tuple[dict, str]:
    """Generate climb/cruise/descent performance estimates using the OpenAP library.

    Simulates a complete flight profile (climb, cruise, descent) for the
    given aircraft type and route distance, computing segment times,
    distances, ground speeds, and fuel burns.  All calculations assume
    zero-wind conditions (TAS equals ground speed).

    Args:
        ac_type: ICAO aircraft type designator (e.g. ``"A320"``, ``"B738"``).
            OpenAP synonym resolution is attempted automatically.
        cruise_alt_ft: Target cruise altitude in feet.
        mass_kg: Aircraft mass in kilograms.  When ``None``, the function
            applies a three-level fallback chain:
            1. 85 % of the aircraft's MTOW from OpenAP property tables.
            2. If MTOW is unavailable, a generic narrow-body default of
               60 000 kg.
            3. If the property lookup itself fails, 60 000 kg.
        route_dist_km: Total great-circle route distance in kilometres, used
            to derive the cruise segment length after subtracting climb and
            descent distances.

    Returns:
        A two-element tuple:
            - A dictionary with keys ``"block"``, ``"climb"``, ``"cruise"``,
              ``"descent"``, and ``"assumptions"`` containing the per-segment
              and total performance estimates.
            - A string identifying the engine backend (``"openap"``).

    Raises:
        OpenAPError: If the OpenAP package is not available.
    """
    if not OPENAP_AVAILABLE:
        raise OpenAPError("OpenAP backend unavailable. Please `pip install openap`.")

    # ------------------------------------------------------------------
    # Mass resolution fallback chain
    # ------------------------------------------------------------------
    # Priority 1: Use the caller-supplied mass_kg if provided.
    # Priority 2: Look up the aircraft's MTOW from OpenAP property tables
    #             and use 85 % of MTOW as a conservative operating mass.
    # Priority 3: If MTOW is not listed, fall back to 60 000 kg -- a
    #             rough generic narrow-body estimate.
    # Priority 4: If the property lookup itself throws, use 60 000 kg.
    mass = mass_kg
    engine_note = "openap"
    try:
        ac_props = prop.aircraft(ac_type, use_synonym=True)
        mtow = (ac_props.get("limits") or {}).get("MTOW") or ac_props.get("mtow")
        if mass is None and mtow:
            mass = 0.85 * float(mtow)  # 85 % MTOW -- conservative default
        elif mass is None:
            mass = 60_000.0  # generic narrow-body fallback
    except Exception:
        # Aircraft type not recognised or property tables unavailable.
        mass = mass or 60_000.0

    fgen = FlightGenerator(ac=ac_type)
    dt = 10  # simulation time-step in seconds

    # Generate climb & descent DataFrames at the requested cruise altitude.
    # Some OpenAP versions do not accept the ``alt_cr`` keyword; fall back
    # to default altitude if a TypeError is raised.
    try:
        climb = fgen.climb(dt=dt, alt_cr=cruise_alt_ft)
    except TypeError:
        climb = fgen.climb(dt=dt)

    try:
        descent = fgen.descent(dt=dt, alt_cr=cruise_alt_ft)
    except TypeError:
        descent = fgen.descent(dt=dt)

    # Cruise segment baseline (used for average speed extraction)
    cruise_seg = fgen.cruise(dt=dt)

    def seg(df):
        """Extract summary statistics from an OpenAP flight-phase DataFrame.

        Reads the last row for cumulative time (``t``) and distance (``s``),
        and computes mean altitude, ground speed, and vertical rate across
        the entire phase.

        Args:
            df: A pandas DataFrame produced by ``FlightGenerator`` methods.

        Returns:
            A tuple of (time_seconds, distance_km, mean_alt_ft,
            mean_groundspeed_kts, mean_vertical_rate_fpm).
        """
        t_s = float(df["t"].iloc[-1])
        dist_km = float(df["s"].iloc[-1]) / 1000.0  # 's' column is in metres
        alt_ft = float(df["altitude"].mean())
        gs_kts = float(df["groundspeed"].mean())
        vs_fpm = float(df["vertical_rate"].mean())
        return t_s, dist_km, alt_ft, gs_kts, vs_fpm

    t_climb, d_climb, a_climb, gs_climb, vs_climb = seg(climb)
    t_des, d_des, a_des, gs_des, vs_des = seg(descent)
    _, _, a_cru, gs_cru, _ = seg(cruise_seg)

    # ------------------------------------------------------------------
    # Cruise time calculation
    # ------------------------------------------------------------------
    # Subtract climb and descent distances from the total route to find the
    # remaining distance to be flown at cruise speed.
    d_remaining = max(0.0, route_dist_km - (d_climb + d_des))

    # NOTE: The first computation below is intentionally left as-is for
    # historical context -- it uses a convoluted unit conversion that
    # produces a correct but hard-to-read expression.  The second (and
    # authoritative) computation replaces it with a clearer approach:
    #   cruise_speed_km_per_s = (gs_cru [kts] * NM_PER_KM [nm/km inverted]) / 3600
    # effectively converting knots -> km/s, then time = distance / speed.
    cruise_time_s = (
        0.0 if gs_cru <= 1e-6 else (d_remaining * KM_PER_NM) / (gs_cru / 3600.0 / 1.852)
    )
    # Corrected, clearer cruise-time computation:
    # kts -> km/s: multiply knots by (KM_PER_NM / 3600) since 1 kt = 1 NM/h.
    # Equivalently: speed_km_s = gs_cru * NM_PER_KM / 3600  (but NM_PER_KM < 1,
    # so we use the reciprocal form below).
    cruise_time_s = (
        0.0 if gs_cru <= 1e-6 else (d_remaining / ((gs_cru * NM_PER_KM) / 3600.0))
    )

    fuelflow = FuelFlow(ac=ac_type)

    def fuel_from(
        avg_gs_kts: float, avg_alt_ft: float, vs_fpm: float, time_s: float
    ) -> float:
        """Estimate fuel burn for a flight phase from average flight parameters.

        Uses the OpenAP ``FuelFlow.enroute`` model.  Under the zero-wind
        assumption, True Airspeed (TAS) is set equal to ground speed (GS).

        Args:
            avg_gs_kts: Average ground speed in knots.
            avg_alt_ft: Average altitude in feet.
            vs_fpm: Average vertical rate in feet per minute.
            time_s: Duration of the phase in seconds.

        Returns:
            Estimated fuel burn in kilograms.
        """
        # Zero-wind assumption: TAS ~ GS for baseline fuel estimation.
        try:
            ff_kg_s = float(
                fuelflow.enroute(mass=mass, tas=avg_gs_kts, alt=avg_alt_ft, vs=vs_fpm)
            )
        except Exception:
            ff_kg_s = 0.0
        return ff_kg_s * time_s

    fuel_climb = fuel_from(gs_climb, a_climb, vs_climb, t_climb)
    fuel_des = fuel_from(gs_des, a_des, vs_des, t_des)
    fuel_cru = fuel_from(gs_cru, cruise_alt_ft, 0.0, cruise_time_s)

    # Build per-segment result models
    climb_out = SegmentEst(
        time_min=t_climb / 60.0,
        distance_km=d_climb,
        avg_gs_kts=gs_climb,
        fuel_kg=fuel_climb,
    )
    cruise_out = SegmentEst(
        time_min=cruise_time_s / 60.0,
        distance_km=d_remaining,
        avg_gs_kts=gs_cru,
        fuel_kg=fuel_cru,
    )
    des_out = SegmentEst(
        time_min=t_des / 60.0, distance_km=d_des, avg_gs_kts=gs_des, fuel_kg=fuel_des
    )

    # Aggregate block (total) figures
    block_time_min = climb_out.time_min + cruise_out.time_min + des_out.time_min
    block_fuel_kg = climb_out.fuel_kg + cruise_out.fuel_kg + des_out.fuel_kg

    estimates = {
        "block": {"time_min": block_time_min, "fuel_kg": block_fuel_kg},
        "climb": climb_out.model_dump(),
        "cruise": cruise_out.model_dump(),
        "descent": des_out.model_dump(),
        "assumptions": {
            "zero_wind": True,  # All fuel/time estimates assume no wind
            "mass_kg": mass,
            "cruise_alt_ft": cruise_alt_ft,
        },
    }
    return estimates, engine_note


# ---------------------------------------------------------------------------
# Business logic functions for MCP
# ---------------------------------------------------------------------------


def health() -> dict:
    """Return a lightweight health-check status for the system.

    Returns:
        A dictionary with keys ``"status"`` (always ``"ok"``),
        ``"openap"`` (boolean indicating OpenAP availability), and
        ``"airports_count"`` (number of IATA entries loaded).
    """
    return {
        "status": "ok",
        "openap": OPENAP_AVAILABLE,
        "airports_count": len(_AIRPORTS_IATA),
    }


def airports_by_city(city: str, country: str | None = None) -> list[AirportOut]:
    """Search for airports by city name with an optional country filter.

    This is a thin wrapper around ``_find_city_airports`` exposed as part
    of the public API for the MCP interface.

    Args:
        city: City name to search for.
        country: Optional ISO alpha-2 country code.

    Returns:
        A list of matching ``AirportOut`` instances, sorted with
        international airports first.
    """
    return _find_city_airports(city, country)


def plan_flight(payload: dict) -> dict:
    """Plan a flight using the provided request payload (dict-based API).

    Accepts a raw dictionary (typically from a JSON request body), validates
    it against ``PlanRequest``, resolves airports, computes the
    great-circle route, generates performance estimates, and returns a
    serialised ``PlanResponse``.

    Args:
        payload: Dictionary whose keys correspond to ``PlanRequest`` fields.

    Returns:
        A dictionary representation of ``PlanResponse``.

    Raises:
        ValueError: If validation fails, airports cannot be resolved, or
            departure and arrival are identical.
        RuntimeError: If the performance backend fails or an unexpected
            error occurs.
    """
    try:
        # Validate the request using PlanRequest model
        req = PlanRequest(**payload)

        # Check for identical departure and arrival
        if (
            req.depart_city.strip().lower() == req.arrive_city.strip().lower()
            and not req.prefer_arrive_iata
            and not req.prefer_depart_iata
        ):
            raise ValueError(
                "Departure and arrival look identical—please specify airports explicitly."
            )

        # Resolve departure and arrival airports
        dep = _resolve_endpoint(
            req.depart_city,
            req.depart_country,
            req.prefer_depart_iata,
            role="departure",
        )
        arr = _resolve_endpoint(
            req.arrive_city, req.arrive_country, req.prefer_arrive_iata, role="arrival"
        )

        # Calculate great-circle route
        poly, dist_km = great_circle_points(
            dep.lat, dep.lon, arr.lat, arr.lon, req.route_step_km
        )

        # Generate estimates
        if req.backend == "openap":
            est, engine_name = estimates_openap(
                req.ac_type, req.cruise_alt_ft, req.mass_kg, dist_km
            )
        else:
            raise ValueError(f"Unknown backend: {req.backend}")

        # Build response
        response = PlanResponse(
            engine=engine_name,
            depart=dep,
            arrive=arr,
            distance_km=dist_km,
            distance_nm=dist_km * NM_PER_KM,
            polyline=poly,
            estimates=est,
        )

        return response.model_dump()

    except (ValueError, AirportResolutionError) as e:
        raise ValueError(str(e))
    except OpenAPError as e:
        raise RuntimeError(str(e))
    except Exception as e:
        raise RuntimeError(f"Flight planning failed: {str(e)}")


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def search_airports_by_city(city: str, country: str | None = None) -> list[AirportOut]:
    """Search for airports by city name with an optional country filter.

    Alias for ``_find_city_airports`` exposed as a public entry-point for
    use by higher-level interface modules (e.g. the FastMCP tool layer).

    Args:
        city: City name to search for.
        country: Optional ISO alpha-2 country code.

    Returns:
        A list of matching ``AirportOut`` instances.
    """
    return _find_city_airports(city, country)


def get_health_status() -> dict:
    """Return an extended health-check status including domain availability.

    Unlike ``health()``, this function also queries the integration layer
    for the availability of optional aerospace domain modules (atmosphere,
    aerodynamics, orbital mechanics, etc.) and includes a version string.

    Returns:
        A dictionary with keys ``"status"``, ``"openap"``,
        ``"airports_count"``, ``"version"``, and ``"domains"``.
    """
    from .integrations import get_domain_status

    domain_status = get_domain_status()

    return {
        "status": "ok",
        "openap": OPENAP_AVAILABLE,
        "airports_count": len(_AIRPORTS_IATA),
        "version": "0.1.0",
        "domains": domain_status,
    }


class FlightPlanError(Exception):
    """Raised when the end-to-end flight planning pipeline fails.

    Wraps lower-level errors (airport resolution, performance estimation)
    into a single exception type for the public ``create_flight_plan`` API.
    """

    pass


def create_flight_plan(req: PlanRequest) -> PlanResponse:
    """Create a complete flight plan from a validated request.

    This is the strongly-typed public API (as opposed to ``plan_flight``
    which accepts a raw dictionary).  It resolves airports, computes the
    great-circle route, generates performance estimates, and returns a
    fully populated ``PlanResponse``.

    Args:
        req: A validated ``PlanRequest`` instance containing route and
            aircraft configuration.

    Returns:
        A ``PlanResponse`` with resolved airports, polyline waypoints,
        and per-segment performance estimates.

    Raises:
        FlightPlanError: If departure and arrival are identical, airport
            resolution fails, or the performance backend encounters an error.
    """
    if (
        req.depart_city.strip().lower() == req.arrive_city.strip().lower()
        and not req.prefer_arrive_iata
        and not req.prefer_depart_iata
    ):
        raise FlightPlanError(
            "Departure and arrival look identical—please specify airports explicitly."
        )

    try:
        dep = _resolve_endpoint(
            req.depart_city,
            req.depart_country,
            req.prefer_depart_iata,
            role="departure",
        )
        arr = _resolve_endpoint(
            req.arrive_city, req.arrive_country, req.prefer_arrive_iata, role="arrival"
        )
    except AirportResolutionError as e:
        raise FlightPlanError(str(e))

    # Great-circle route
    poly, dist_km = great_circle_points(
        dep.lat, dep.lon, arr.lat, arr.lon, req.route_step_km
    )

    # Performance estimates via the selected backend
    if req.backend == "openap":
        try:
            est, engine_name = estimates_openap(
                req.ac_type, req.cruise_alt_ft, req.mass_kg, dist_km
            )
        except OpenAPError as e:
            raise FlightPlanError(str(e))
    else:
        raise FlightPlanError(f"Unknown backend: {req.backend}")

    return PlanResponse(
        engine=engine_name,
        depart=dep,
        arrive=arr,
        distance_km=dist_km,
        distance_nm=dist_km * NM_PER_KM,
        polyline=poly,
        estimates=est,
    )
