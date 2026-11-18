"""ROI polygon helpers used by the detector to filter detections."""
import pytest

from yarnbreak.roi import _parse_roi, _point_in_poly, _roi_top_split_y

SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]


def test_parse_polygon():
    assert _parse_roi([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]) == SQUARE


def test_parse_flat_rectangle_sorts_corners():
    assert _parse_roi([0.8, 0.8, 0.2, 0.2]) == SQUARE


@pytest.mark.parametrize("roi", [None, [], [0, 0, 1, 1], [[0, 0], [1, 1]], ["a", "b"]])
def test_parse_rejects_full_frame_and_invalid(roi):
    assert _parse_roi(roi) is None


@pytest.mark.parametrize("point, inside", [
    ((0.5, 0.5), True),
    ((0.1, 0.5), False),
    ((0.5, 0.9), False),
])
def test_point_in_poly(point, inside):
    assert _point_in_poly(*point, SQUARE) is inside


def test_point_in_triangle():
    tri = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    assert _point_in_poly(0.2, 0.2, tri)
    assert not _point_in_poly(0.8, 0.8, tri)


def test_top_split_y():
    assert _roi_top_split_y(SQUARE, 0.10) == pytest.approx(0.26)
    assert _roi_top_split_y(SQUARE, 0.0) is None
    assert _roi_top_split_y([], 0.1) is None
