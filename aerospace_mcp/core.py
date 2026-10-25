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

import copy
import functools
import importlib.util
import logging
import math
import re
import unicodedata
from typing import Any, Literal, NamedTuple

import airportsdata
from geographiclib.geodesic import Geodesic
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Optional / lazy import for OpenAP (perf + fuel)
# ---------------------------------------------------------------------------
# OpenAP may not be installed in every deployment (e.g. lightweight containers
# or serverless runtimes).  Importing it costs ~1.2 s (pandas + YAML model
# files), so it is deferred until the first performance estimate.  The
# availability flag is derived from the import system without importing the
# package; if a later import fails anyway, ``_load_openap`` flips it to False.
OPENAP_AVAILABLE = importlib.util.find_spec("openap") is not None

# Names published as module attributes once OpenAP has been imported.  They
# stay reachable as ``aerospace_mcp.core.FlightGenerator`` etc. (resolved on
# first access by the module ``__getattr__`` below), so ``unittest.mock.patch``
# targets such as ``"aerospace_mcp.core.FuelFlow"`` keep working.
_OPENAP_EXPORTS = ("FuelFlow", "FlightGenerator", "prop")


def _load_openap() -> None:
    """Import OpenAP and publish its entry points as module globals.

    Names that already exist in the module namespace (e.g. a test double
    installed with ``mock.patch``) are left untouched.

    Raises:
        ImportError: If OpenAP is not installed or fails to import; the
            ``OPENAP_AVAILABLE`` flag is cleared before re-raising.
    """
    global OPENAP_AVAILABLE
    try:
        from openap import FuelFlow, prop
        from openap.gen import FlightGenerator
    except Exception:
        OPENAP_AVAILABLE = False
        raise
    namespace = globals()
    namespace.setdefault("FuelFlow", FuelFlow)
    namespace.setdefault("FlightGenerator", FlightGenerator)
    namespace.setdefault("prop", prop)


def __getattr__(name: str) -> Any:
    """Resolve the lazily imported OpenAP names on first attribute access."""
    if name in _OPENAP_EXPORTS:
        try:
            _load_openap()
        except Exception as e:
            raise AttributeError(f"OpenAP is not available: {e}") from e
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _openap(name: str) -> Any:
    """Return an OpenAP entry point by name, importing OpenAP on first use."""
    if name not in globals():
        _load_openap()
    return globals()[name]


# ---------------------------------------------------------------------------
# Unit-conversion constants
# ---------------------------------------------------------------------------
# 1 nautical mile (NM) = 1.852 km exactly, by international definition.
# NM_PER_KM: multiply a distance in kilometres to get nautical miles.
NM_PER_KM = 0.539956803  # 1 / 1.852
# KM_PER_NM: multiply a distance in nautical miles to get kilometres.
KM_PER_NM = 1.0 / NM_PER_KM  # ~1.852

logger = logging.getLogger(__name__)


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
    depart_city: str = Field(
        ..., min_length=1, max_length=100, description="e.g., 'San Jose'"
    )
    arrive_city: str = Field(
        ..., min_length=1, max_length=100, description="e.g., 'Tokyo'"
    )
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
        None, gt=0, description="If not set, defaults to 85% MTOW when available"
    )
    route_step_km: float = Field(
        25.0, gt=1.0, le=1000.0, description="Sampling step for polyline points"
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
    return _airport_out(iata, ap)


# ---------------------------------------------------------------------------
# City search index and ranking
# ---------------------------------------------------------------------------
# Precomputed structures for fast city lookups, built lazily on first search
# and rebuilt automatically whenever ``_AIRPORTS_IATA`` is swapped out (tests
# patch it with fixture data).  The identity check makes patching transparent:
# a patched dict is a different object, so the index is rebuilt from it.
_CITY_INDEX: dict[str, list[str]] | None = None  # folded city -> [iata, ...]
_NAME_LIST: list[tuple[str, str]] | None = None  # (iata, folded name)
_INDEX_SOURCE: object | None = None

# Busy passenger airports, ordered by approximate annual passenger traffic
# (ACI World rankings, 2019/2023, rounded).  Only used as a ranking signal
# between airports that already match the same query, so the order matters
# mostly *within* a metro area (LHR > LGW > STN > LTN > LCY, HND > NRT,
# CDG > ORY, ...) and between same-named cities in different countries
# (FCO for "Rome", not Rome NY).  The list is deliberately coarse.
_MAJOR_AIRPORTS_BY_TRAFFIC: tuple[str, ...] = tuple(
    """
    ATL DXB DFW LHR HND DEN IST LAX ORD DEL CDG JFK
    CAN AMS FRA PVG SIN ICN PEK MAD BKK CGK SFO LAS
    SEA MCO MIA CLT PHX EWR BCN HKG BOM SZX CTU KUL
    IAH YYZ FCO DOH MEX SYD MUC LGW MNL BOS MSP SVO
    SAW GRU NRT LGA DTW PHL FLL BLR SLC BWI IAD DCA
    ZRH LIS TPE KMG SHA PKX MEL DUB VIE CPH MXP YVR
    CUN BOG JED RUH AUH MAN ORY SAN KIX STN OSL ARN
    ATH PMI HEL BRU DME GMP SGN HAN DPS BNE AKL PER
    LIM SCL EZE GIG JNB CAI YUL BER WAW PRG LTN TLV
    ITM DMK HYD MAA CCU MDW HOU DAL VKO CGH NCE LCY
    """.split()
)
_TRAFFIC_RANK: dict[str, int] = {
    code: rank for rank, code in enumerate(_MAJOR_AIRPORTS_BY_TRAFFIC)
}

# Main airports whose ``city`` field in the airport database differs from
# the metro name people search for (e.g. YYZ is listed under Mississauga,
# EWR under Newark, DFW under "Dallas-Fort Worth").  They are treated as
# exact-city matches for the metro name.
_METRO_AIRPORTS: dict[str, tuple[str, ...]] = {
    "new york": ("EWR",),
    "toronto": ("YYZ",),
    "dallas": ("DFW",),
    "fort worth": ("DFW",),
    "buenos aires": ("EZE",),
    "rome": ("CIA",),
    "taipei": ("TSA",),
}


def _fold(text: str) -> str:
    """Normalise text for matching: strip accents, lower-case, squash spaces.

    ``"Malé"`` and ``"male"`` fold to the same key, so accented city names
    in the database match plain-ASCII queries.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.lower().split())


def _get_city_indexes() -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Return (city index, folded name list), rebuilding if the DB changed.

    Both structures skip entries without a valid IATA code (heliports,
    closed fields, etc.) so search never has to re-filter them.
    """
    global _CITY_INDEX, _NAME_LIST, _INDEX_SOURCE
    if _INDEX_SOURCE is not _AIRPORTS_IATA:
        city_index: dict[str, list[str]] = {}
        name_list: list[tuple[str, str]] = []
        for iata, ap in _AIRPORTS_IATA.items():
            if not iata or not ap.get("iata"):
                continue
            city_index.setdefault(_fold(ap.get("city", "")), []).append(iata)
            name_list.append((iata, _fold(ap.get("name", ""))))
        _CITY_INDEX = city_index
        _NAME_LIST = name_list
        _INDEX_SOURCE = _AIRPORTS_IATA
    assert _CITY_INDEX is not None and _NAME_LIST is not None
    return _CITY_INDEX, _NAME_LIST


def _airport_out(iata: str, ap: dict) -> AirportOut:
    """Build an ``AirportOut`` from a raw ``airportsdata`` record."""
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

    Matching is case- and accent-insensitive and works in tiers; a lower
    tier is consulted only when every higher tier is empty (after the
    country filter):

    1. **Exact city** -- the airport's ``city`` field equals the query, or
       the airport is a known main airport of that metro area even though
       the database files it under a suburb (``_METRO_AIRPORTS``, e.g.
       YYZ for "Toronto", EWR for "New York").
    2. **City word match** -- the query appears as whole words inside the
       city field ("Delhi" -> "New Delhi"), followed by **name word
       match** -- the query appears as whole words in the airport name
       ("Heathrow" -> "London Heathrow Airport").  Substrings inside a
       word never match, so "Nice" no longer finds "Eunice".

    Within a tier, airports are ranked deterministically by:

    a. approximate passenger traffic for busy airports
       (``_MAJOR_AIRPORTS_BY_TRAFFIC``; others rank after all of them),
    b. "International" in the name,
    c. the query appearing in the airport name (the city's namesake field,
       e.g. "Norman Y Mineta San Jose International" for "San Jose"),
    d. airport name, then IATA code, alphabetically.

    Example: "San Jose" without a country matches SJC (US), SJO and SYQ
    (CR) and RHV (US) exactly; none is in the traffic list and the first
    three are all "International", so SJC wins on rule (c).  SJC also has
    the most passenger traffic of the three.  Pass ``country="CR"`` to get
    SJO.

    Args:
        city: City name to search for (e.g. ``"Tokyo"``).
        country: Optional ISO alpha-2 country code (e.g. ``"JP"``) to narrow
            results.  When ``None``, all countries are searched.

    Returns:
        A list of ``AirportOut`` instances matching the query, best match
        first.  May be empty if no matches are found.
    """
    query = _fold(city)
    if not query:  # Return empty list for empty city names
        return []

    city_index, name_list = _get_city_indexes()
    country_u = country.strip().upper() if country else None

    def keep(iata: str) -> bool:
        ap = _AIRPORTS_IATA.get(iata)
        if not ap:
            return False
        return country_u is None or ap.get("country", "").upper() == country_u

    # Tier 1: exact city (plus metro-area main airports).
    exact = dict.fromkeys([*city_index.get(query, []), *_METRO_AIRPORTS.get(query, ())])
    tiers: list[list[str]] = [[iata for iata in exact if keep(iata)]]

    if not tiers[0]:
        # Tier 2: whole-word matches in the city field, then in the name.
        word = re.compile(rf"(?<!\w){re.escape(query)}(?!\w)")
        city_words = [
            iata
            for city_key, codes in city_index.items()
            if query in city_key and word.search(city_key)
            for iata in codes
            if keep(iata)
        ]
        seen = set(city_words)
        name_words = [
            iata
            for iata, name_f in name_list
            if iata not in seen and query in name_f and word.search(name_f)
            if keep(iata)
        ]
        tiers = [city_words, name_words]

    no_rank = len(_MAJOR_AIRPORTS_BY_TRAFFIC)

    def rank(iata: str) -> tuple:
        ap = _AIRPORTS_IATA[iata]
        name_f = _fold(ap.get("name", ""))
        return (
            _TRAFFIC_RANK.get(iata.upper(), no_rank),
            "international" not in name_f,
            query not in name_f,
            ap.get("name", ""),
            iata,
        )

    out: list[AirportOut] = []
    for tier in tiers:
        for iata in sorted(dict.fromkeys(tier), key=rank):
            out.append(_airport_out(iata, _AIRPORTS_IATA[iata]))
    return out


class AirportResolutionError(Exception):
    """Raised when an airport cannot be resolved from the given input.

    This may happen if an explicit IATA code is invalid or if no airports
    match a city/country query.

    Attributes:
        kind: Machine-readable failure category -- ``"iata_not_found"`` or
            ``"city_not_found"`` -- so interface layers can map to
            appropriate status codes without parsing the message.
    """

    def __init__(self, message: str, kind: str = "city_not_found") -> None:
        super().__init__(message)
        self.kind = kind


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
            raise AirportResolutionError(
                f"{role}: IATA '{prefer_iata}' not found.", kind="iata_not_found"
            )
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

    Attributes:
        kind: Machine-readable failure category so interface layers can map
            it to a status code without parsing the message:
            ``"unavailable"`` (OpenAP not installed / not importable),
            ``"unsupported_aircraft"`` (type not covered by OpenAP's
            models), or ``"model_error"`` (a model evaluation failed).
    """

    def __init__(self, message: str, kind: str = "model_error") -> None:
        super().__init__(message)
        self.kind = kind


# Simulation time step for OpenAP's trajectory generator, in seconds.
_SIM_DT_S = 10
# Lowest top-of-climb used when a short route forces a capped profile.
_MIN_TOP_OF_CLIMB_FT = 1000.0

# ---------------------------------------------------------------------------
# OpenAP object / profile caches
# ---------------------------------------------------------------------------
# Constructing OpenAP models re-parses their YAML data files on every call
# (FuelFlow ~15 ms, FlightGenerator ~5 ms, prop.aircraft ~3 ms each), and the
# climb/descent simulations take ~5-6 ms each.  None of that depends on the
# route, so it is computed once per aircraft type (and cruise altitude) and
# cached.  The factory / lookup callable is part of every cache key, so a
# patched ``FlightGenerator``, ``FuelFlow`` or ``prop.aircraft`` (tests) never
# receives objects cached from the real one.  Cached values are treated as
# read-only: profiles are stored as non-writeable numpy arrays and aircraft
# property dicts are deep-copied before being handed out.


class _Profile(NamedTuple):
    """Read-only time histories extracted from an OpenAP flight phase.

    Attributes:
        t: Elapsed time in seconds (starts at 0).
        s: Ground distance in metres (starts at 0).
        alt: Altitude in feet.
        gs: Ground speed (= TAS, zero wind) in knots.
        vs: Vertical rate in feet per minute.
    """

    t: Any
    s: Any
    alt: Any
    gs: Any
    vs: Any

    @classmethod
    def from_frame(cls, df: Any) -> _Profile:
        """Copy the relevant DataFrame columns into non-writeable arrays."""
        import numpy as np

        arrays = []
        for col in ("t", "s", "altitude", "groundspeed", "vertical_rate"):
            arr = np.array(df[col], dtype=float)
            arr.setflags(write=False)
            arrays.append(arr)
        return cls(*arrays)

    def summary(self) -> tuple[float, float, float, float, float]:
        """Return (time_s, distance_km, mean_alt_ft, mean_gs_kts, mean_vs_fpm).

        Reads the last sample for cumulative time and distance and averages
        altitude, ground speed and vertical rate across the phase.
        """
        return (
            float(self.t[-1]),
            float(self.s[-1]) / 1000.0,  # 's' is in metres
            float(self.alt.mean()),
            float(self.gs.mean()),
            float(self.vs.mean()),
        )


@functools.lru_cache(maxsize=64)
def _cached_model(factory: Any, ac_type: str) -> Any:
    """Build an OpenAP model (``FlightGenerator`` / ``FuelFlow``) once per type.

    ``use_synonym=True`` lets OpenAP substitute the closest modelled type for
    aircraft it has property data for but no kinematic/fuel model (A20N ->
    A320, B38M -> B737, A359 -> B789, ...).  Exceptions are not cached.
    """
    return factory(ac=ac_type, use_synonym=True)


@functools.lru_cache(maxsize=64)
def _cached_aircraft_props(lookup: Any, ac_type: str) -> dict:
    """Return ``prop.aircraft(ac_type, use_synonym=True)``, parsed once."""
    return lookup(ac_type, use_synonym=True)


def _aircraft_props(ac_type: str) -> dict:
    """Return a private copy of OpenAP's property dict for ``ac_type``."""
    return copy.deepcopy(_cached_aircraft_props(_openap("prop").aircraft, ac_type))


@functools.lru_cache(maxsize=256)
def _cached_profile(fgen: Any, phase: str, alt_cr_ft: int, dt: int) -> _Profile:
    """Simulate a climb or descent once per (generator, altitude, step).

    Some OpenAP versions do not accept the ``alt_cr`` keyword; fall back to
    the default altitude if a TypeError is raised.
    """
    method = getattr(fgen, phase)
    try:
        frame = method(dt=dt, alt_cr=alt_cr_ft)
    except TypeError:
        frame = method(dt=dt)
    return _Profile.from_frame(frame)


@functools.lru_cache(maxsize=64)
def _cached_cruise_mach(fgen: Any) -> float | None:
    """Default cruise Mach from OpenAP's WRAP model, or None if unavailable."""
    try:
        return float(fgen.wrap.cruise_mach()["default"])
    except Exception:
        return None


def _openap_aero() -> Any:
    """OpenAP's unit/ISA helper module (``openap.aero`` since 2.5, earlier
    ``openap.extra.aero``); fresh installs get the newest openap, so both
    locations are supported."""
    try:
        from openap import aero
    except ImportError:
        from openap.extra import aero
    return aero


def _cruise_tas_kts(fgen: Any, cruise_alt_ft: float, dt: int) -> float:
    """True airspeed in knots at the WRAP default cruise Mach and altitude.

    TAS = M * a(h), with the ISA speed of sound at the *requested* cruise
    altitude (OpenAP's own ``cruise()`` generator would fly its default
    altitude instead).
    """
    mach = _cached_cruise_mach(fgen)
    if mach is not None:
        aero = _openap_aero()
        return float(aero.mach2tas(mach, cruise_alt_ft * aero.ft) / aero.kts)
    # Fallback for generators without a WRAP cruise-Mach accessor: simulate
    # a level segment at the requested altitude and average its speed.
    frame = fgen.cruise(dt=dt, alt_cr=cruise_alt_ft)
    return float(frame["groundspeed"].mean())


def _fit_profiles_to_route(
    climb: _Profile, descent: _Profile, route_km: float
) -> tuple[_Profile, _Profile, float | None, float | None, float]:
    """Cap the climb/descent profiles so they fit inside a short route.

    When the full climb to and descent from cruise altitude cover more
    ground than the route itself, the profile is cut at the highest
    top-of-climb altitude whose climb distance plus the descent distance
    from that same altitude still fits.  The climb is truncated at that
    altitude and the descent starts there, so the leftover distance (if
    any) is flown level at the top-of-climb altitude and speed.  If even a
    ``_MIN_TOP_OF_CLIMB_FT`` profile is too long (or the route has zero
    length), that lowest profile is scaled (distance and time alike, so
    speeds are preserved) to exactly the route length.

    Args:
        climb: Full climb profile to the requested cruise altitude.
        descent: Full descent profile from the requested cruise altitude.
        route_km: Route distance in kilometres.

    Returns:
        ``(climb, descent, top_alt_ft, level_gs_kts, scale)``.
        ``top_alt_ft`` / ``level_gs_kts`` are None when no capping was
        needed; ``scale`` is 1.0 unless the scaling fallback was used.
    """
    import numpy as np

    route_m = route_km * 1000.0
    if climb.s[-1] + descent.s[-1] <= route_m:
        return climb, descent, None, None, 1.0

    # j[i]: first descent sample at or below the altitude of climb sample i
    # (descent altitude is non-increasing, so search the negated array).
    j = np.searchsorted(-descent.alt, -climb.alt, side="left")
    j = np.minimum(j, len(descent.s) - 1)
    ground_m = climb.s + (descent.s[-1] - descent.s[j])

    feasible = np.flatnonzero(
        (ground_m <= route_m) & (climb.alt >= _MIN_TOP_OF_CLIMB_FT)
    )
    scale = 1.0
    if feasible.size:
        i = int(feasible[-1])
    else:
        airborne = np.flatnonzero(climb.alt >= _MIN_TOP_OF_CLIMB_FT)
        i = int(airborne[0]) if airborne.size else len(climb.s) - 1
        scale = route_m / ground_m[i] if ground_m[i] > 0 else 0.0
    k = int(j[i])

    capped_climb = _Profile(
        climb.t[: i + 1] * scale,
        climb.s[: i + 1] * scale,
        climb.alt[: i + 1],
        climb.gs[: i + 1],
        climb.vs[: i + 1],
    )
    capped_descent = _Profile(
        (descent.t[k:] - descent.t[k]) * scale,
        (descent.s[k:] - descent.s[k]) * scale,
        descent.alt[k:],
        descent.gs[k:],
        descent.vs[k:],
    )
    return capped_climb, capped_descent, float(climb.alt[i]), float(climb.gs[i]), scale


def estimates_openap(
    ac_type: str,
    cruise_alt_ft: int,
    mass_kg: float | None,
    route_dist_km: float,
    headwind_kts: float = 0.0,
) -> tuple[dict, str]:
    """Generate climb/cruise/descent performance estimates using the OpenAP library.

    Simulates a complete flight profile (climb, cruise, descent) for the
    given aircraft type and route distance, computing segment times,
    distances, ground speeds, and fuel burns.  By default all calculations
    assume zero-wind conditions (TAS equals ground speed); a constant
    cruise headwind component can be supplied to adjust cruise ground
    speed, time, and fuel.

    Cruise is flown at OpenAP's default cruise Mach for the type at the
    requested ``cruise_alt_ft`` (TAS = M * a(h)).  Routes shorter than the
    full climb + descent are flown with a capped top of climb (see
    ``_fit_profiles_to_route``) so the three segments always add up to the
    route distance; ``assumptions.top_of_climb_ft`` and
    ``assumptions.profile_note`` report the cap.

    Args:
        ac_type: ICAO aircraft type designator (e.g. ``"A320"``, ``"B738"``).
            OpenAP synonym resolution is used, so types without their own
            kinematic/fuel model (``"A20N"``, ``"B38M"``, ...) are modelled
            with OpenAP's closest equivalent.
        cruise_alt_ft: Target cruise altitude in feet.
        mass_kg: Aircraft mass in kilograms (must be > 0).  When ``None``,
            the function applies a fallback chain:
            1. 85 % of the aircraft's MTOW from OpenAP property tables.
            2. If MTOW or the property lookup is unavailable, a generic
               narrow-body default of 60 000 kg.
        route_dist_km: Total great-circle route distance in kilometres, used
            to derive the cruise segment length after subtracting climb and
            descent distances.
        headwind_kts: Average headwind component along the cruise segment in
            knots.  Positive values (headwind) reduce ground speed and
            increase time/fuel; negative values (tailwind) do the opposite.
            Cruise ground speed is floored at 30 kts to avoid divide-by-zero
            for unrealistically strong winds.  Climb/descent segments are
            left unadjusted (they are short and speed-scheduled).

    Returns:
        A two-element tuple:
            - A dictionary with keys ``"block"``, ``"climb"``, ``"cruise"``,
              ``"descent"``, and ``"assumptions"`` containing the per-segment
              and total performance estimates.
            - A string identifying the engine backend (``"openap"``).

    Raises:
        ValueError: If ``mass_kg`` is given and not positive.
        OpenAPError: If the OpenAP package is not available, the aircraft
            type is not supported by OpenAP's models, or the fuel-flow
            model fails.
    """
    if not OPENAP_AVAILABLE:
        raise OpenAPError(
            "OpenAP backend unavailable. Please `pip install openap`.",
            kind="unavailable",
        )
    if mass_kg is not None and not mass_kg > 0:
        raise ValueError(f"mass_kg must be > 0 (got {mass_kg})")
    try:
        flight_generator_cls = _openap("FlightGenerator")
        fuel_flow_cls = _openap("FuelFlow")
    except Exception as e:
        raise OpenAPError(
            f"OpenAP backend unavailable ({e}). Please `pip install openap`.",
            kind="unavailable",
        ) from e

    # ------------------------------------------------------------------
    # Mass resolution fallback chain
    # ------------------------------------------------------------------
    # Priority 1: Use the caller-supplied mass_kg if provided.
    # Priority 2: Look up the aircraft's MTOW from OpenAP property tables
    #             and use 85 % of MTOW as a conservative operating mass.
    # Priority 3: If MTOW is not listed or the lookup fails, fall back to
    #             60 000 kg -- a rough generic narrow-body estimate.
    mass = mass_kg
    mass_source = "user" if mass_kg is not None else "default_60t"
    engine_note = "openap"
    try:
        ac_props = _aircraft_props(ac_type)
        mtow = (ac_props.get("limits") or {}).get("MTOW") or ac_props.get("mtow")
        if mass is None and mtow:
            mass = 0.85 * float(mtow)  # 85 % MTOW -- conservative default
            mass_source = "mtow_85pct"
    except Exception:
        # Aircraft type not recognised or property tables unavailable.
        pass
    if mass is None:
        mass = 60_000.0  # generic narrow-body fallback
    if mass_source == "default_60t":
        logger.warning(
            "No MTOW available for aircraft type %r; using generic 60t mass default",
            ac_type,
        )

    # OpenAP raises plain ValueError for aircraft its kinematic/fuel models
    # don't cover; surface that as a clean domain error instead of a 500.
    try:
        fgen = _cached_model(flight_generator_cls, ac_type)
    except Exception as e:
        raise OpenAPError(
            f"Aircraft type '{ac_type}' not supported by OpenAP: {e}",
            kind="unsupported_aircraft",
        ) from e
    dt = _SIM_DT_S

    try:
        climb = _cached_profile(fgen, "climb", cruise_alt_ft, dt)
        descent = _cached_profile(fgen, "descent", cruise_alt_ft, dt)
        gs_cru = _cruise_tas_kts(fgen, cruise_alt_ft, dt)
    except Exception as e:
        raise OpenAPError(
            f"OpenAP trajectory generation failed for {ac_type}: {e}"
        ) from e

    # Short routes: cap the top of climb so the profile fits the route.
    climb, descent, top_alt_ft, level_gs_kts, scale = _fit_profiles_to_route(
        climb, descent, route_dist_km
    )
    cruise_level_ft = float(cruise_alt_ft) if top_alt_ft is None else top_alt_ft
    if level_gs_kts is not None:
        gs_cru = level_gs_kts

    t_climb, d_climb, a_climb, gs_climb, vs_climb = climb.summary()
    t_des, d_des, a_des, gs_des, vs_des = descent.summary()

    # ------------------------------------------------------------------
    # Cruise time calculation
    # ------------------------------------------------------------------
    # Subtract climb and descent distances from the total route to find the
    # remaining distance to be flown at cruise speed.
    d_remaining = max(0.0, route_dist_km - (d_climb + d_des))

    # Apply the cruise headwind component: ground speed = TAS - headwind.
    # TAS itself (used for fuel flow) is unchanged by wind.
    gs_cru_eff = gs_cru if headwind_kts == 0.0 else max(gs_cru - headwind_kts, 30.0)

    # kts -> km/s: 1 kt = 1 NM/h, so speed_km_s = gs * KM_PER_NM / 3600.
    cruise_time_s = (
        0.0 if gs_cru_eff <= 1e-6 else d_remaining / ((gs_cru_eff * KM_PER_NM) / 3600.0)
    )

    try:
        fuelflow = _cached_model(fuel_flow_cls, ac_type)
    except Exception as e:
        raise OpenAPError(
            f"Aircraft type '{ac_type}' not supported by OpenAP fuel model: {e}",
            kind="unsupported_aircraft",
        ) from e

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

        Raises:
            OpenAPError: If the fuel-flow model fails, rather than silently
                reporting zero fuel burn.
        """
        # Zero-wind assumption: TAS ~ GS for baseline fuel estimation.
        try:
            ff_kg_s = float(
                fuelflow.enroute(mass=mass, tas=avg_gs_kts, alt=avg_alt_ft, vs=vs_fpm)
            )
        except Exception as e:
            raise OpenAPError(f"Fuel flow computation failed for {ac_type}: {e}") from e
        return ff_kg_s * time_s

    fuel_climb = fuel_from(gs_climb, a_climb, vs_climb, t_climb)
    fuel_des = fuel_from(gs_des, a_des, vs_des, t_des)
    fuel_cru = fuel_from(gs_cru, cruise_level_ft, 0.0, cruise_time_s)

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
        avg_gs_kts=gs_cru_eff,
        fuel_kg=fuel_cru,
    )
    des_out = SegmentEst(
        time_min=t_des / 60.0, distance_km=d_des, avg_gs_kts=gs_des, fuel_kg=fuel_des
    )

    # Aggregate block (total) figures
    block_time_min = climb_out.time_min + cruise_out.time_min + des_out.time_min
    block_fuel_kg = climb_out.fuel_kg + cruise_out.fuel_kg + des_out.fuel_kg

    assumptions: dict[str, Any] = {
        "zero_wind": headwind_kts == 0.0,
        "headwind_kts": headwind_kts,
        "mass_kg": mass,
        "mass_source": mass_source,
        "cruise_alt_ft": cruise_alt_ft,
        "top_of_climb_ft": cruise_level_ft,
    }
    if top_alt_ft is not None:
        full_km = (
            _cached_profile(fgen, "climb", cruise_alt_ft, dt).s[-1]
            + _cached_profile(fgen, "descent", cruise_alt_ft, dt).s[-1]
        ) / 1000.0
        note = (
            f"Route ({route_dist_km:.1f} km) is shorter than a full climb to and "
            f"descent from {cruise_alt_ft} ft ({full_km:.1f} km); top of climb "
            f"capped at {top_alt_ft:.0f} ft."
        )
        if scale != 1.0:
            note += (
                f" Even that profile is too long, so climb and descent distances"
                f" and times were scaled by {scale:.3f} to fit the route."
            )
        assumptions["profile_note"] = note

    estimates = {
        "block": {"time_min": block_time_min, "fuel_kg": block_fuel_kg},
        "climb": climb_out.model_dump(),
        "cruise": cruise_out.model_dump(),
        "descent": des_out.model_dump(),
        "assumptions": assumptions,
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
        # Validate the request, then run the shared single-leg pipeline.
        req = PlanRequest(**payload)
        return _plan_single_leg(req).model_dump()
    except (ValueError, AirportResolutionError, FlightPlanError) as e:
        raise ValueError(str(e)) from e
    except OpenAPError as e:
        raise RuntimeError(str(e)) from e
    except Exception as e:
        raise RuntimeError(f"Flight planning failed: {str(e)}") from e


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
        FlightPlanError: If departure and arrival are identical or the
            backend is unknown.
        AirportResolutionError: If either endpoint cannot be resolved.
        OpenAPError: If the performance backend is unavailable or fails.
    """
    return _plan_single_leg(req)


def _plan_single_leg(req: PlanRequest) -> PlanResponse:
    """Resolve, route and estimate one leg (shared by the public planners).

    Identical endpoints are detected on the *resolved* airports, so the
    same city name in two countries ("San Jose" CR -> "San Jose" US) is a
    valid request, while two inputs that land on the same airport (same
    city twice, or the same IATA code twice) are rejected.

    Raises:
        FlightPlanError: If both endpoints resolve to the same airport or
            the backend is unknown.
        AirportResolutionError: If either endpoint cannot be resolved.
        OpenAPError: If the performance backend is unavailable or fails.
    """
    dep = _resolve_endpoint(
        req.depart_city,
        req.depart_country,
        req.prefer_depart_iata,
        role="departure",
    )
    arr = _resolve_endpoint(
        req.arrive_city, req.arrive_country, req.prefer_arrive_iata, role="arrival"
    )
    if dep.iata == arr.iata:
        raise FlightPlanError(
            f"Departure and arrival are identical: both resolve to {dep.iata} "
            f"({dep.name}). Please specify different airports."
        )

    # Great-circle route
    poly, dist_km = great_circle_points(
        dep.lat, dep.lon, arr.lat, arr.lon, req.route_step_km
    )

    # Performance estimates via the selected backend
    if req.backend == "openap":
        est, engine_name = estimates_openap(
            req.ac_type, req.cruise_alt_ft, req.mass_kg, dist_km
        )
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


MAX_MULTI_LEG_WAYPOINTS = 10


def plan_multi_leg(
    waypoints: list[dict],
    ac_type: str,
    cruise_alt_ft: int = 35000,
    mass_kg: float | None = None,
    route_step_km: float = 25.0,
) -> dict:
    """Plan a multi-leg journey through an ordered list of waypoints.

    Each consecutive pair of waypoints becomes one leg, planned with the
    same airport resolution, great-circle routing, and OpenAP performance
    estimation as a single-leg plan.  Totals are aggregated across legs.

    Args:
        waypoints: Ordered list of 2-10 waypoint dicts, each with keys
            ``city`` (required unless ``iata`` given), optional ``country``
            (ISO alpha-2), and optional ``iata`` (forces a specific airport).
        ac_type: ICAO aircraft type designator used for every leg.
        cruise_alt_ft: Cruise altitude in feet for every leg.
        mass_kg: Aircraft mass override in kg (85% MTOW default when None).
            Note: mass is NOT depleted between legs; each leg uses the same
            starting mass, so total fuel is a conservative estimate.
        route_step_km: Polyline sampling interval per leg in kilometres.

    Returns:
        A dictionary with ``legs`` (list of per-leg summaries including
        airports, distance, and estimates) and ``totals`` (aggregate
        distance, time, and fuel).

    Raises:
        FlightPlanError: If the waypoint list or an aircraft parameter is
            invalid (the same bounds as ``PlanRequest`` apply).
        AirportResolutionError: If any waypoint cannot be resolved.
        OpenAPError: If the performance backend is unavailable or fails.
    """
    # Validate the aircraft knobs against the single-leg PlanRequest bounds
    # (cruise 8000-45000 ft, 1 < step <= 1000 km, mass > 0) so both planners
    # accept exactly the same inputs.
    try:
        knobs = PlanRequest(
            depart_city="-",
            arrive_city="-",
            ac_type=ac_type,
            cruise_alt_ft=cruise_alt_ft,
            mass_kg=mass_kg,
            route_step_km=route_step_km,
        )
    except ValidationError as e:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        raise FlightPlanError(f"invalid aircraft parameters: {problems}") from e
    # Continue with the validated (type-coerced) values.
    ac_type, cruise_alt_ft = knobs.ac_type, knobs.cruise_alt_ft
    mass_kg, route_step_km = knobs.mass_kg, knobs.route_step_km

    if not isinstance(waypoints, list) or len(waypoints) < 2:
        raise FlightPlanError("waypoints must be a list of at least 2 entries")
    if len(waypoints) > MAX_MULTI_LEG_WAYPOINTS:
        raise FlightPlanError(
            f"waypoints must contain at most {MAX_MULTI_LEG_WAYPOINTS} entries"
        )

    # Resolve every waypoint up-front so errors name the failing stop
    # before any expensive per-leg work happens.
    resolved: list[AirportOut] = []
    for idx, wp in enumerate(waypoints):
        if not isinstance(wp, dict):
            raise FlightPlanError(f"waypoint {idx} must be an object/dict")
        city = str(wp.get("city", "") or "")
        country = wp.get("country")
        iata = wp.get("iata")
        if not city and not iata:
            raise FlightPlanError(f"waypoint {idx} needs a 'city' or 'iata' key")
        resolved.append(_resolve_endpoint(city, country, iata, role=f"waypoint {idx}"))

    legs = []
    total_distance_km = 0.0
    total_time_min = 0.0
    total_fuel_kg = 0.0

    for i in range(len(resolved) - 1):
        dep, arr = resolved[i], resolved[i + 1]
        if dep.iata == arr.iata:
            raise FlightPlanError(
                f"leg {i + 1}: departure and arrival are both {dep.iata}"
            )

        poly, dist_km = great_circle_points(
            dep.lat, dep.lon, arr.lat, arr.lon, route_step_km
        )
        est, engine_name = estimates_openap(ac_type, cruise_alt_ft, mass_kg, dist_km)

        total_distance_km += dist_km
        total_time_min += est["block"]["time_min"]
        total_fuel_kg += est["block"]["fuel_kg"]

        legs.append(
            {
                "leg": i + 1,
                "depart": dep.model_dump(),
                "arrive": arr.model_dump(),
                "distance_km": dist_km,
                "distance_nm": dist_km * NM_PER_KM,
                "polyline": poly,
                "estimates": est,
                "engine": engine_name,
            }
        )

    return {
        "legs": legs,
        "totals": {
            "legs": len(legs),
            "distance_km": total_distance_km,
            "distance_nm": total_distance_km * NM_PER_KM,
            "time_min": total_time_min,
            "fuel_kg": total_fuel_kg,
        },
        "assumptions": {
            "constant_mass_between_legs": True,
            "zero_wind": True,
            "cruise_alt_ft": cruise_alt_ft,
            "ac_type": ac_type,
        },
    }
