"""Distance-based channel partner matching (Intelligence Engine only).

Tiers (explicit, always one of):
- NEAREST_BY_DISTANCE: user coordinates AND partner coordinates both
  available; partners sorted by computed Haversine distance with the
  distance value included. The ONLY tier allowed to use "nearest".
- EXACT_DISTRICT: no coordinates usable, but a partner of the right
  type is listed in the user's district.
- STATE_LEVEL: no district match, but a partner of the right type
  exists in the user's state.
- TYPE_ONLY_NO_LISTING: no partner found; the required partner type
  is returned as guidance instead of "unavailable" + nothing.

Rules (never violated):
- Never invent a partner or a coordinate; unknown stays unknown.
- Missing coordinates fall back to district/state tiers, never error.
- Partner availability NEVER affects eligibility results — this module
  does not import or call the eligibility engine.
- Pure Python, no external API, no paid maps dependency.
"""

import math

from intelligence_engine.schemas import (
    ChannelPartner,
    PartnerMatch,
    PartnerMatchResult,
    PartnerType,
)

_EARTH_RADIUS_KM = 6371.0088


def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in kilometres between two lat/long pairs.

    Pure function (Haversine formula, mean Earth radius 6371.0088 km).
    No external API. Inputs are decimal degrees.
    """
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))
    hav = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


def _norm(value: str | None) -> str | None:
    """Normalized comparison key; None stays None."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    return cleaned or None


class PartnerMatcher:
    """Tiered matcher over ChannelPartner listings (info only)."""

    def match(
        self,
        required_partner_type: PartnerType,
        partners: list[ChannelPartner],
        user_state: str | None,
        user_district: str | None = None,
        user_latitude: float | None = None,
        user_longitude: float | None = None,
    ) -> PartnerMatchResult:
        """Match partners of the required type with an explicit tier.

        Args:
            required_partner_type: Scheme's required partner type.
            partners: Candidate listings (never fabricated here).
            user_state: User's state (optional but needed below distance).
            user_district: User's district (optional).
            user_latitude/user_longitude: User coordinates (optional).

        Returns:
            PartnerMatchResult with an explicit tier. Never errors on
            missing coordinates; never claims "nearest" without an
            actually computed distance; never returns "unavailable"
            with nothing (TYPE_ONLY_NO_LISTING echoes the type).
        """
        typed = [p for p in partners if p.partner_type == required_partner_type]
        user_has_coords = user_latitude is not None and user_longitude is not None
        geo = [p for p in typed if p.latitude is not None and p.longitude is not None]

        # Tier 1: true distance ranking (both sides have coordinates).
        if user_has_coords and geo:
            assert user_latitude is not None and user_longitude is not None
            ranked = sorted(
                (
                    PartnerMatch(
                        partner=p,
                        distance_km=round(
                            haversine_distance_km(
                                user_latitude, user_longitude, p.latitude, p.longitude  # type: ignore[arg-type]
                            ),
                            3,
                        ),
                    )
                    for p in geo
                ),
                key=lambda m: (m.distance_km if m.distance_km is not None else float("inf"), m.partner.id),
            )
            return PartnerMatchResult(
                tier="NEAREST_BY_DISTANCE",
                required_partner_type=required_partner_type,
                matches=ranked,
                reasons=[
                    f"Nearest {required_partner_type} partner(s) by computed Haversine distance "
                    f"({len(ranked)} listed with coordinates)."
                ],
            )

        # Tier 2: district listing fallback.
        wanted_district = _norm(user_district)
        wanted_state = _norm(user_state)
        if wanted_district is not None:
            district_hits = [
                PartnerMatch(partner=p, distance_km=None)
                for p in sorted(typed, key=lambda p: p.id)
                if _norm(p.district) == wanted_district
            ]
            if district_hits:
                return PartnerMatchResult(
                    tier="EXACT_DISTRICT",
                    required_partner_type=required_partner_type,
                    matches=district_hits,
                    reasons=[
                        f"{len(district_hits)} {required_partner_type} partner(s) listed in district "
                        f"'{user_district}' (no distance computed; coordinates unavailable)."
                    ],
                )

        # Tier 3: state listing fallback.
        if wanted_state is not None:
            state_hits = [
                PartnerMatch(partner=p, distance_km=None)
                for p in sorted(typed, key=lambda p: p.id)
                if _norm(p.state) == wanted_state
            ]
            if state_hits:
                return PartnerMatchResult(
                    tier="STATE_LEVEL",
                    required_partner_type=required_partner_type,
                    matches=state_hits,
                    reasons=[
                        f"{len(state_hits)} {required_partner_type} partner(s) listed in state "
                        f"'{user_state}' (no district match; no distance computed)."
                    ],
                )

        # Tier 4: guidance only — never bare "unavailable".
        return PartnerMatchResult(
            tier="TYPE_ONLY_NO_LISTING",
            required_partner_type=required_partner_type,
            matches=[],
            reasons=[
                f"No {required_partner_type} partner listing found for the given location; "
                f"approach any {required_partner_type} channel partner for this scheme."
            ],
        )
