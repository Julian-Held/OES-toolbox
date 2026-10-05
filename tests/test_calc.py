import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from OES_toolbox.calc import scale_wavelengths


@pytest.mark.parametrize(
    "first_order, expected",
    [(1, [1, 2, 3, 4, 5]), (2, [-1, 1, 3, 5, 7]), (0.5,[2,2.5,3,3.5,4])],
    ids=["identity", "stretch", "compress"],
)
def test_scale_wavelengths(first_order, expected):
    x = np.array([1, 2, 3, 4, 5])
    expected = np.array(expected)
    result = scale_wavelengths(x, first_order=first_order)
    assert_equal(result, expected)
    assert result.mean() == expected.mean()


@pytest.mark.parametrize(
    "x",
    [
        np.array([1, 2, 3, 4, 5]),
        np.array([1, 2, 3, 4]),
        np.array([1, 2, 4, 8]),
        np.array([8.5, 4.25, 2.0, 1.5]),
        np.array([1, 2, 4, 8], dtype=np.float32),
        np.array([3]),
    ],
    ids=["odd", "even", "nonuniform", "descending", "float32", "single element"],
)
@pytest.mark.parametrize("first_order", [-1, 0, 0.5, 1, 2])
def test_scale_wavelengths_first_order_preserves_center(x, first_order):
    center_index = x.size // 2
    center = x[center_index]
    original = x.copy()
    result = scale_wavelengths(x, first_order=first_order)
    expected = center + first_order * (x - center)
    assert_allclose(result, expected)
    assert_equal(result[center_index], center)
    assert_equal(x, original)


@pytest.mark.parametrize(
    "values, first_order, second_order, expected",
    [
        ([1, 2, 3, 4, 5], 1, 0.5, [3, 2.5, 3, 4.5, 7]),
        ([1, 2, 3, 4, 5], 1, -0.5, [-1, 1.5, 3, 3.5, 3]),
        ([1, 2, 3, 4, 5], 0, 0.5, [5, 3.5, 3, 3.5, 5]),
        ([1, 2, 3, 4], 0.5, 0.1, [2.4, 2.6, 3, 3.6]),
        ([1, 2, 4, 8], 0.5, 0.1, [3.4, 3.4, 4, 7.6]),
        ([8, 4, 2, 1], 0.5, 0.1, [8.6, 3.4, 2, 1.6]),
        ([3], 0.5, 0.1, [3]),
    ],
    ids=["positive-curvature", "negative-curvature", "pure-quadratic", "even", "nonuniform", "descending", "single element"],
)
def test_scale_wavelengths_quadratic_correction(values, first_order, second_order, expected):
    x = np.array(values)
    original = x.copy()
    result = scale_wavelengths(x, first_order=first_order, second_order=second_order)
    assert_allclose(result, expected)
    assert_equal(result[x.size // 2], x[x.size // 2])
    assert_equal(x, original)