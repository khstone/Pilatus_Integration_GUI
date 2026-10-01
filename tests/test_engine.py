"""Reduction regression tests on one real scan (May 2026, SSRL BL2-1, CaCO3 + CuO, scan 1).

The reference .xye was produced at the time with the 0519 calibration; the Poisson path must
keep reproducing it.
"""
import numpy as np
import pytest

import Integration_engine as eng
from conftest import CAL, IMAGES, REF, SETTINGS, SPEC, USER


@pytest.fixture(scope="module")
def xyz():
    db, R = eng.Read_Cal(CAL)
    return eng.make_map(db, R)


@pytest.fixture(scope="module")
def poisson(xyz):
    return eng.IntegrationEngine().integrate(SPEC, 1, IMAGES, USER, xyz, dict(SETTINGS))


@pytest.fixture(scope="module")
def azimuthal(xyz):
    return eng.IntegrationEngine().integrate_var(SPEC, 1, IMAGES, USER, xyz, dict(SETTINGS))


def test_poisson_reproduces_reference_xye(poisson):
    name, x, y, e = poisson
    ref = np.loadtxt(REF)
    assert name == "KHS10_27D_CaCO3_CuO_anneal_scan1.xye"
    assert len(x) == len(ref) and np.allclose(x, ref[:, 0])
    assert np.allclose(y, ref[:, 1], rtol=1e-6)
    assert np.allclose(e, ref[:, 2], rtol=1e-6)


def test_azimuthal_grid_matches_poisson(poisson, azimuthal):
    assert np.allclose(azimuthal[1], poisson[1]), "2theta axis must be the same in both error models"


def test_azimuthal_intensity_matches_poisson(poisson, azimuthal):
    """Both modes give the mean pixel value per bin (times the same I0 scale)."""
    _, x, yp, _ = poisson
    _, xa, ya, _ = azimuthal
    assert np.allclose(ya, yp, rtol=1e-6, atol=1e-9)


def test_azimuthal_error_is_pixel_spread(xyz, azimuthal):
    """sigma = standard deviation of the I0-normalized pixel values in each bin (x I0 scale),
    computed here independently for a few bins."""
    _, x, y, e = azimuthal
    tth, i0 = eng.SPECread(SPEC, 1)
    step = float(SETTINGS["stepsize"])
    vals = {}
    targets = x[[200, 2000, 5000]]
    for k in range(len(tth)):
        fn = f"{IMAGES}/{USER}_KHS10_27D_CaCO3_CuO_anneal_scan1_{k:04d}.raw"
        data = eng.read_RAW(fn, SETTINGS["img_clip_low"], SETTINGS["img_clip_high"])
        t = eng.cart2sphere(eng.rotate_operation(xyz, float(tth[k]))).flatten()
        v = data.flatten() / i0[k]
        lab = np.round((np.floor(t / step) + 1) * step, 3)       # bin label = upper edge
        for tgt in targets:
            sel = (np.abs(lab - tgt) < step / 2) & (v >= 0)
            vals.setdefault(round(float(tgt), 3), []).append(v[sel])
    mult = float(i0[0])
    for tgt, chunks in vals.items():
        allv = np.concatenate(chunks)
        i = int(np.argmin(np.abs(x - tgt)))
        assert np.isclose(e[i], mult * np.std(allv, ddof=1), rtol=1e-6), tgt
        assert np.isclose(y[i], mult * np.mean(allv), rtol=1e-6), tgt


def test_stepsize_that_does_not_divide_180(xyz):
    s = dict(SETTINGS, stepsize="0.007")
    name, x, y, e = eng.IntegrationEngine().integrate(SPEC, 1, IMAGES, USER, xyz, s)
    assert np.allclose(np.diff(x), 0.007, atol=1e-9), "bins must have exactly the requested width"
