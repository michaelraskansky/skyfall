"""Unit tests for ingestion/satcat_lookup.py SatcatLookup."""

from ingestion.satcat_lookup import SatcatInfo, SatcatLookup


class TestSatcatInfo:
    """Tests for the SatcatInfo model."""

    def test_defaults(self):
        info = SatcatInfo(norad_cat_id="53493")
        assert info.object_name == "UNKNOWN"
        assert info.country == "UNKNOWN"
        assert info.launch_date == ""
        assert info.object_type == ""
        assert info.rcs_size == ""

    def test_full_construction(self):
        info = SatcatInfo(
            norad_cat_id="53493",
            object_name="STARLINK-4361",
            country="US",
            launch_date="2022-08-12",
            object_type="PAYLOAD",
            rcs_size="LARGE",
        )
        assert info.object_name == "STARLINK-4361"
        assert info.country == "US"
        assert info.launch_date == "2022-08-12"
        assert info.object_type == "PAYLOAD"
        assert info.rcs_size == "LARGE"

    def test_serializable(self):
        info = SatcatInfo(norad_cat_id="53493", object_name="TEST")
        d = info.model_dump()
        assert d["norad_cat_id"] == "53493"
        assert d["object_name"] == "TEST"


class TestSatcatLookupCache:
    """Tests for in-memory cache behavior."""

    def test_cache_starts_empty(self):
        lookup = SatcatLookup()
        assert len(lookup._cache) == 0

    def test_cache_stores_and_retrieves(self):
        lookup = SatcatLookup()
        info = SatcatInfo(norad_cat_id="12345", object_name="CACHED-SAT")
        lookup._cache["12345"] = info
        assert lookup._cache["12345"].object_name == "CACHED-SAT"

    def test_cache_keyed_by_norad_id(self):
        lookup = SatcatLookup()
        lookup._cache["11111"] = SatcatInfo(norad_cat_id="11111", object_name="SAT-A")
        lookup._cache["22222"] = SatcatInfo(norad_cat_id="22222", object_name="SAT-B")
        assert lookup._cache["11111"].object_name == "SAT-A"
        assert lookup._cache["22222"].object_name == "SAT-B"
        assert "33333" not in lookup._cache
