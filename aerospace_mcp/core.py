from __future__ import annotations

import math
from typing import Literal

import airportsdata
from geographiclib.geodesic import Geodesic
from pydantic import BaseModel, Field

# Optional / graceful import for OpenAP (perf + fuel)
OPENAP_AVAILABLE = True
try:
    from openap import FuelFlow, prop
    from openap.gen import FlightGenerator
except Exception:
    OPENAP_AVAILABLE = False

# Constants
NM_PER_KM = 0.539956803
KM_PER_NM = 1.0 / NM_PER_KM


# ----------------------------
# Models
# ----------------------------
class AirportOut(BaseModel):
    iata: str
    icao: str
    name: str
    city: str
    country: str
    lat: float
    lon: float
    tz: str | None = None


class PlanRequest(BaseModel):
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
    time_min: float
    distance_km: float
    avg_gs_kts: float
    fuel_kg: float


class PlanResponse(BaseModel):
    engine: str
    depart: AirportOut
    arrive: AirportOut
    distance_km: float
    distance_nm: float
    polyline: list[tuple[float, float]]  # [(lat, lon), ...]
    estimates: dict  # {"block": {...}, "climb": {...}, ...}


# ----------------------------
# Airport data loading
# ----------------------------
_AIRPORTS_IATA = airportsdata.load("IATA")  # Fast, in-process dict (no network)
# Shape example: {'SJC': {'iata':'SJC','icao':'KSJC','name':'San Jose Intl', 'city':'San Jose', 'country':'US', 'lat':..., 'lon':...}, ...}


# ----------------------------
# Airport resolution functions
# ----------------------------
def _airport_from_iata(iata: str) -> AirportOut | None:
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
    city_l = city.strip().lower()
    out = []
    for iata, ap in _AIRPORTS_IATA.items():
        if not iata or not ap.get("iata"):
            continue
        if (
            ap.get("city", "").strip().lower() == city_l
            or city_l in ap.get("name", "").lower()
        ):
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
    # Heuristic: prefer airports with "International" in the name, else keep order
    out.sort(key=lambda a: ("international" not in a.name.lower(), a.name))
    # De-dup city matches that are clearly heliports or without IATA (already filtered)
    return out


class AirportResolutionError(Exception):
    """Raised when airport resolution fails"""

    pass


def _resolve_endpoint(
    city: str,
    country: str | None,
    prefer_iata: str | None,
    role: str,
) -> AirportOut:
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
    return cands[0]


# ----------------------------
# Geodesic / polyline
# ----------------------------
def great_circle_points(
    lat1: float, lon1: float, lat2: float, lon2: float, step_km: float
) -> tuple[list[tuple[float, float]], float]:
    g = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
    dist_m = g["s12"]
    line = Geodesic.WGS84.Line(lat1, lon1, g["azi1"])
    n = max(1, int(math.ceil((dist_m / 1000.0) / step_km)))
    pts = []
    for i in range(n + 1):
        s = min(dist_m, (dist_m * i) / n)
        p = line.Position(s)
        pts.append((p["lat2"], p["lon2"]))
    return pts, dist_m / 1000.0


# ----------------------------
# OpenAP estimates (climb / cruise / descent)
# ----------------------------
class OpenAPError(Exception):
    """Raised when OpenAP operations fail"""

    pass


def estimates_openap(
    ac_type: str, cruise_alt_ft: int, mass_kg: float | None, route_dist_km: float
) -> tuple[dict, str]:
    if not OPENAP_AVAILABLE:
        raise OpenAPError("OpenAP backend unavailable. Please `pip install openap`.")

    # Resolve mass default from aircraft properties (fallback to 85% MTOW if present)
    mass = mass_kg
    engine_note = "openap"
    try:
        ac_props = prop.aircraft(ac_type, use_synonym=True)
        mtow = (ac_props.get("limits") or {}).get("MTOW") or ac_props.get("mtow")
        if mass is None and mtow:
            mass = 0.85 * float(mtow)  # conservative default
        elif mass is None:
            # fallback generic narrowbody guess
            mass = 60_000.0
    except Exception:
        mass = mass or 60_000.0

    fgen = FlightGenerator(ac=ac_type)
    dt = 10  # seconds

    # Generate climb & descent segments at requested cruise altitude (if allowed)
    try:
        climb = fgen.climb(dt=dt, alt_cr=cruise_alt_ft)
    except TypeError:
        climb = fgen.climb(dt=dt)

    try:
        descent = fgen.descent(dt=dt, alt_cr=cruise_alt_ft)
    except TypeError:
        descent = fgen.descent(dt=dt)

    # Cruise segment baseline
    cruise_seg = fgen.cruise(dt=dt)

    def seg(df):
        t_s = float(df["t"].iloc[-1])
        dist_km = float(df["s"].iloc[-1]) / 1000.0  # 's' is meters in OpenAP docs
        alt_ft = float(df["altitude"].mean())
        gs_kts = float(df["groundspeed"].mean())
        vs_fpm = float(df["vertical_rate"].mean())
        return t_s, dist_km, alt_ft, gs_kts, vs_fpm

    t_climb, d_climb, a_climb, gs_climb, vs_climb = seg(climb)
    t_des, d_des, a_des, gs_des, vs_des = seg(descent)
    _, _, a_cru, gs_cru, _ = seg(cruise_seg)

    # How much cruise distance remains after climb+descent?
    d_remaining = max(0.0, route_dist_km - (d_climb + d_des))
    # Guard for super-short hops
    cruise_time_s = (
        0.0 if gs_cru <= 1e-6 else (d_remaining * KM_PER_NM) / (gs_cru / 3600.0 / 1.852)
    )  # but simpler to compute by kts:
    # Convert properly: kts = nm/hour → km/s = (kts * NM_PER_KM) / 3600
    cruise_time_s = (
        0.0 if gs_cru <= 1e-6 else (d_remaining / ((gs_cru * NM_PER_KM) / 3600.0))
    )

    fuelflow = FuelFlow(ac=ac_type)

    def fuel_from(
        avg_gs_kts: float, avg_alt_ft: float, vs_fpm: float, time_s: float
    ) -> float:
        # TAS ~ GS (zero-wind assumption for baseline)
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

    # Build segments
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

    block_time_min = climb_out.time_min + cruise_out.time_min + des_out.time_min
    block_fuel_kg = climb_out.fuel_kg + cruise_out.fuel_kg + des_out.fuel_kg

    estimates = {
        "block": {"time_min": block_time_min, "fuel_kg": block_fuel_kg},
        "climb": climb_out.model_dump(),
        "cruise": cruise_out.model_dump(),
        "descent": des_out.model_dump(),
        "assumptions": {
            "zero_wind": True,
            "mass_kg": mass,
            "cruise_alt_ft": cruise_alt_ft,
        },
    }
    return estimates, engine_note


# ----------------------------
# Business logic functions for MCP
# ----------------------------
def health() -> dict:
    """Health check endpoint - returns system status."""
    return {
        "status": "ok",
        "openap": OPENAP_AVAILABLE,
        "airports_count": len(_AIRPORTS_IATA),
    }


def airports_by_city(city: str, country: str | None = None) -> list[AirportOut]:
    """Search for airports by city and optional country."""
    return _find_city_airports(city, country)


def plan_flight(payload: dict) -> dict:
    """Plan a flight using the provided request payload."""
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


# ----------------------------
# Public API functions
# ----------------------------
def search_airports_by_city(city: str, country: str | None = None) -> list[AirportOut]:
    """Search for airports by city name."""
    return _find_city_airports(city, country)


def get_health_status() -> dict:
    """Get system health status."""
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
    """Raised when flight planning fails"""

    pass


def create_flight_plan(req: PlanRequest) -> PlanResponse:
    """Create a flight plan from the request."""
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

    # Estimates
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
