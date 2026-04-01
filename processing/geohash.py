"""Pure-Python geohash encoder with neighbor computation.

Implements the standard geohash algorithm (base-32) for mapping lat/lon
coordinates to grid cells, plus neighbor lookup for boundary queries.

Reference: https://en.wikipedia.org/wiki/Geohash
"""

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_NEIGHBORS = {
    "n": {"even": "p0r21436x8zb9dcf5h7kjnmqesgutwvy", "odd": "bc01fg45238967deuvhjyznpkmstqrwx"},
    "s": {"even": "14365h7k9dcfesgujnmqp0r2twvyx8zb", "odd": "238967debc01fg45uvhjyznpkmstqrwx"},
    "e": {"even": "bc01fg45238967deuvhjyznpkmstqrwx", "odd": "p0r21436x8zb9dcf5h7kjnmqesgutwvy"},
    "w": {"even": "238967debc01fg45uvhjyznpkmstqrwx", "odd": "14365h7k9dcfesgujnmqp0r2twvyx8zb"},
}

_BORDERS = {
    "n": {"even": "prxz", "odd": "bcfguvyz"},
    "s": {"even": "028b", "odd": "0145hjnp"},
    "e": {"even": "bcfguvyz", "odd": "prxz"},
    "w": {"even": "0145hjnp", "odd": "028b"},
}


def encode(latitude: float, longitude: float, precision: int = 4) -> str:
    """Encode a latitude/longitude pair into a geohash string.

    Args:
        latitude: Latitude in degrees (-90 to 90).
        longitude: Longitude in degrees (-180 to 180).
        precision: Number of characters in the resulting geohash (default 4).

    Returns:
        Geohash string of the given precision.
    """
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude must be in [-90, 90], got {latitude}")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude must be in [-180, 180], got {longitude}")
    if precision < 1:
        raise ValueError(f"precision must be >= 1, got {precision}")

    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)
    is_lon = True  # longitude bit comes first
    bit = 0
    ch_index = 0
    result = []

    while len(result) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2
            if longitude >= mid:
                ch_index = ch_index * 2 + 1
                lon_range = (mid, lon_range[1])
            else:
                ch_index = ch_index * 2
                lon_range = (lon_range[0], mid)
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if latitude >= mid:
                ch_index = ch_index * 2 + 1
                lat_range = (mid, lat_range[1])
            else:
                ch_index = ch_index * 2
                lat_range = (lat_range[0], mid)

        is_lon = not is_lon
        bit += 1

        if bit == 5:
            result.append(_BASE32[ch_index])
            bit = 0
            ch_index = 0

    return "".join(result)


def _adjacent(geohash: str, direction: str) -> str:
    """Return the geohash of the adjacent cell in the given direction.

    Args:
        geohash: Input geohash string.
        direction: One of 'n', 's', 'e', 'w'.

    Returns:
        Geohash string of the neighboring cell.
    """
    last_char = geohash[-1]
    parent = geohash[:-1]
    # In the geohash neighbor algorithm, 'parity' refers to the encoding
    # dimension, not length parity. Even-length hashes end on a latitude bit
    # (odd parity), odd-length hashes end on a longitude bit (even parity).
    parity = "odd" if len(geohash) % 2 == 0 else "even"

    if last_char in _BORDERS[direction][parity] and parent:
        parent = _adjacent(parent, direction)

    return parent + _BASE32[_NEIGHBORS[direction][parity].index(last_char)]


def neighbors(geohash: str) -> list[str]:
    """Return the 8 surrounding geohash cells.

    Args:
        geohash: Input geohash string.

    Returns:
        List of 8 geohash strings representing the surrounding cells
        (N, NE, E, SE, S, SW, W, NW).
    """
    n = _adjacent(geohash, "n")
    s = _adjacent(geohash, "s")
    e = _adjacent(geohash, "e")
    w = _adjacent(geohash, "w")
    ne = _adjacent(n, "e")
    nw = _adjacent(n, "w")
    se = _adjacent(s, "e")
    sw = _adjacent(s, "w")
    return [n, ne, e, se, s, sw, w, nw]
