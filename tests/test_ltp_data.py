"""原始数据换算及 READ 窗口回归。"""

import numpy as np
import pytest

from fit_ltp import prepare, read_trend_residual


def test_prepare_preserves_source_and_signed_conversion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data/ltp_ltd"
    source.mkdir(parents=True)
    a = np.array(
        [
            [0.1e-6, 1.2, 0.01],
            [1.2e-6, -0.4, -0.002],
            [1.3e-6, -0.38, -0.006],
            [2.1e-6, -1.3, -0.01],
            [3.2e-6, -0.4, -0.004],
        ]
    )
    path = source / "ltp_ltd_100ns.csv"
    np.savetxt(path, a, delimiter=",")
    original = path.read_bytes()
    out, g, t, v = prepare(100, tmp_path / "result")
    assert path.read_bytes() == original
    assert g[0] == pytest.approx((-0.004 / 50) / (-0.39 + 0.004))
    assert g[1] == pytest.approx((-0.004 / 50) / (-0.4 + 0.004))
    assert np.all(np.diff(t) > 0)
    assert min(v) < 0 < max(v)
    assert (
        (out / "prepared.csv")
        .read_text()
        .startswith("source_row,time_s,vforce_v,vsense_v")
    )


def test_first_read_error_is_not_hidden_by_bin_median():
    observed = np.ones(12)
    prediction = observed.copy()
    prediction[0] = 3
    groups = [np.arange(6), np.arange(6, 12)]
    assert np.median(prediction[:6]) == np.median(observed[:6])
    assert (
        np.linalg.norm(read_trend_residual(prediction, observed, groups, np.ones(2)))
        > 0
    )
    assert np.all(read_trend_residual(observed, observed, groups, np.ones(2)) == 0)


def test_early_read_error_is_not_hidden_by_trend_bin():
    observed = np.ones(100)
    prediction = observed.copy()
    prediction[4] = 2
    groups = [np.arange(50), np.arange(50, 100)]
    assert np.median(prediction[:50]) == 1
    assert (
        np.linalg.norm(read_trend_residual(prediction, observed, groups, np.ones(2)))
        > 0
    )


def test_initial_width_decoding_preserves_material_geometry():
    from fit_ltp import decode_ltp
    from rram_model import Parameters, State

    reference = Parameters()
    coordinates = [2.0, 0.8, np.log(2e5), 0.5, 0.6, 0.6, 0.5]
    parameters, initial = decode_ltp(coordinates, reference, State())
    assert initial.w_m == pytest.approx(0.5e-9)
    assert parameters.seed_diameter_m == reference.seed_diameter_m
    assert parameters.filament_max_m == reference.filament_max_m
    coordinates[6] = 0.4
    with pytest.raises(ValueError):
        decode_ltp(coordinates, reference, State())


@pytest.mark.parametrize("width", [100, 1])
def test_independent_warm_starts_and_common_width_seeds(width):
    from fit_ltp import (
        INITIAL_WIDTH_BOUNDS_NM,
        decode_ltp,
        load_initialization,
        width_seeds,
    )
    from rram_model import Parameters, State

    warm = load_initialization(width)
    assert f"/{width}ns/" in warm["warm_origin"]
    if width == 1:
        assert "thin_filament" in warm["warm_origin"]
        assert warm["initial_state"]["w_m"] * 1e9 == pytest.approx(0.829, abs=0.002)
    else:
        assert "mean_read_updated" in warm["warm_origin"]
    z = np.array([2, 1, 12, 0.5, 0.6, 0.6, warm["initial_state"]["w_m"] * 1e9])
    assert INITIAL_WIDTH_BOUNDS_NM == (0.5, 2.5)
    for diameter in INITIAL_WIDTH_BOUNDS_NM:
        trial = z.copy()
        trial[6] = diameter
        parameters, initial = decode_ltp(
            trial,
            Parameters(**warm["parameters"]),
            State(**warm["initial_state"]),
        )
        assert initial.w_m * 1e9 == pytest.approx(diameter)
        assert parameters.filament_max_m == 5e-9
    seeds = width_seeds(z)
    np.testing.assert_array_equal(seeds[0], z)
    assert [seed[6] for seed in seeds[1:]] == [0.65, 0.8, 1, 1.5, 2.5]
    for seed in seeds:
        np.testing.assert_array_equal(seed[:6], z[:6])


def test_unified_cli_runs_two_independent_fits(tmp_path, monkeypatch):
    import fit_ltp

    calls = []
    monkeypatch.setattr(fit_ltp, "fit", lambda *args: calls.append(args))
    monkeypatch.setattr(fit_ltp, "summarize", lambda *args: None)
    fit_ltp.main(["--unified", "--max-nfev", "40", "--output", str(tmp_path)])
    assert [call[0] for call in calls] == [100, 1]
    for call in calls:
        assert call[2:] == (20,)
