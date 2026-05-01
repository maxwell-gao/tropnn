from __future__ import annotations


def test_top_level_layer_imports_remain_available() -> None:
    from tropnn import PairwiseLinear, TropFanLinear, TropFanZeroDenseLinear, TropLinear, TropZeroDenseLinear

    assert PairwiseLinear.__name__ == "PairwiseLinear"
    assert TropLinear.__name__ == "TropLinear"
    assert TropZeroDenseLinear.__name__ == "TropZeroDenseLinear"
    assert TropFanLinear.__name__ == "TropFanLinear"
    assert TropFanZeroDenseLinear.__name__ == "TropFanZeroDenseLinear"


def test_layers_package_imports_remain_available() -> None:
    from tropnn.layers import PairwiseLinear, TropFanLinear, TropFanZeroDenseLinear, TropLinear, TropZeroDenseLinear

    assert PairwiseLinear.__name__ == "PairwiseLinear"
    assert TropLinear.__name__ == "TropLinear"
    assert TropZeroDenseLinear.__name__ == "TropZeroDenseLinear"
    assert TropFanLinear.__name__ == "TropFanLinear"
    assert TropFanZeroDenseLinear.__name__ == "TropFanZeroDenseLinear"


def test_experimental_imports_research_variants() -> None:
    from tropnn.experimental import FAN_RECOVERY_MODES, FAN_VALUE_MODES, TropFanLinear, TropFanZeroDenseLinear, TropZeroDenseLinear

    assert TropZeroDenseLinear.__name__ == "TropZeroDenseLinear"
    assert TropFanLinear.__name__ == "TropFanLinear"
    assert TropFanZeroDenseLinear.__name__ == "TropFanZeroDenseLinear"
    assert FAN_VALUE_MODES == ("site", "basis")
    assert FAN_RECOVERY_MODES == ("untied", "tied")


def test_cli_compatibility_modules_import() -> None:
    from tropnn.tools import benchmark, profile, scaling_benchmark

    assert callable(benchmark.main)
    assert callable(profile.main)
    assert callable(scaling_benchmark.main)
