"""ROI polygon geometry helpers.

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple


logger = logging.getLogger("YarnBreak")


def _parse_roi(roi) -> Optional[List[Tuple[float, float]]]:
    """Chuẩn hoá cấu hình ROI về danh sách đỉnh đa giác [(x,y), …] tỉ lệ 0–1.

    Chấp nhận 2 dạng:
      • Đa giác: [[x1,y1], [x2,y2], …]  (≥3 đỉnh, theo thứ tự vòng)
      • Chữ nhật: [x1, y1, x2, y2]       (4 số phẳng → tự dựng 4 góc)
    Trả về None nếu không hợp lệ hoặc bằng toàn khung (không cần lọc)."""
    if not roi:
        return None
    try:
        # Đa giác: mọi phần tử là cặp [x, y].
        if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in roi):
            pts = [(float(x), float(y)) for x, y in roi]
            if len(pts) < 3:
                return None
        # Chữ nhật phẳng [x1,y1,x2,y2].
        elif len(roi) == 4 and all(isinstance(v, (int, float)) for v in roi):
            x1, y1, x2, y2 = (float(v) for v in roi)
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if (x1, y1, x2, y2) == (0.0, 0.0, 1.0, 1.0):
                return None
            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        else:
            return None
    except (ValueError, TypeError):
        return None
    return pts


def _point_in_poly(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting: True nếu điểm (x,y) nằm trong đa giác poly [(px,py), …]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and \
           x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _roi_top_split_y(poly: List[Tuple[float, float]], frac: float) -> Optional[float]:
    """Trả về toạ độ y (tỉ lệ 0–1) chia ROI: mọi điểm có y ≤ giá trị này thuộc
    'vùng trên' (frac% chiều cao ROI tính từ cạnh trên). None nếu không áp dụng."""
    if not poly or frac <= 0.0:
        return None
    ys = [y for _, y in poly]
    ymin, ymax = min(ys), max(ys)
    if ymax <= ymin:
        return None
    return ymin + frac * (ymax - ymin)


def _clip_poly_half(poly: List[Tuple[float, float]], y_line: float,
                    keep_top: bool) -> List[Tuple[float, float]]:
    """Cắt đa giác bằng đường ngang y=y_line (Sutherland–Hodgman). keep_top=True
    giữ phần y ≤ y_line (phía trên), False giữ phần y ≥ y_line (phía dưới)."""
    inside = (lambda p: p[1] <= y_line) if keep_top else (lambda p: p[1] >= y_line)
    out: List[Tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prv = poly[i - 1]
        cur_in, prv_in = inside(cur), inside(prv)
        if cur_in:
            if not prv_in:
                out.append(_intersect_y(prv, cur, y_line))
            out.append(cur)
        elif prv_in:
            out.append(_intersect_y(prv, cur, y_line))
    return out


def _intersect_y(a: Tuple[float, float], b: Tuple[float, float],
                 y_line: float) -> Tuple[float, float]:
    """Giao điểm của đoạn a–b với đường ngang y=y_line."""
    ax, ay = a
    bx, by = b
    if by == ay:
        return (ax, y_line)
    t = (y_line - ay) / (by - ay)
    return (ax + t * (bx - ax), y_line)


# Chiều cao khung GỐC camera (px) mà toạ độ tỉ lệ ROI quy chiếu tới — dùng để đổi
# "nâng góc phải bao nhiêu px" sang tỉ lệ 0–1 (khớp hệ đo của min_box_top_width).
_ROI_REF_HEIGHT_PX = 2048.0


def _right_edge_point(poly: List[Tuple[float, float]], y0: float,
                      drop: float) -> Optional[Tuple[float, float]]:
    """Điểm nằm TRÊN cạnh phải ROI (cạnh cắt đường y0 tại x lớn nhất), trượt DỌC theo
    cạnh xuống 'drop' theo phương y (dương = xuống, âm = lên). Nhờ bám cạnh gốc nên
    điểm không bao giờ lệch ra ngoài ROI."""
    best = None  # (x_tại_y0, đỉnh_a, đỉnh_b)
    n = len(poly)
    for i in range(n):
        a, b = poly[i - 1], poly[i]
        ay, by = a[1], b[1]
        if (ay > y0) != (by > y0):          # cạnh a–b cắt ngang y0
            t = (y0 - ay) / (by - ay)
            x = a[0] + t * (b[0] - a[0])
            if best is None or x > best[0]:
                best = (x, a, b)
    if best is None:
        return None
    _, a, b = best
    yt = y0 + drop
    t = (yt - a[1]) / (b[1] - a[1])
    return (a[0] + t * (b[0] - a[0]), yt)


def _replace_right_y0_vertex(band: List[Tuple[float, float]], y0: float,
                             newpt: Tuple[float, float], eps: float = 1e-6) -> None:
    """Thay đỉnh có x LỚN NHẤT nằm trên đường chia (y≈y0) của 'band' bằng newpt. Dùng
    cho cả band trên lẫn dưới để 2 band vẫn dùng chung đúng một điểm → không hở/chồng."""
    idxs = [i for i, (_, y) in enumerate(band) if abs(y - y0) < eps]
    if not idxs:
        return
    ri = max(idxs, key=lambda i: band[i][0])
    band[ri] = newpt


def _roi_split_polys(poly: Optional[List[Tuple[float, float]]], frac: float,
                     right_drop: float = 0.0):
    """Chia ROI thành (đa giác VÙNG TRÊN, đa giác VÙNG DƯỚI) theo tỉ lệ frac.
    right_drop ≠ 0 → TRƯỢT riêng góc dưới-PHẢI của vùng trên DỌC theo cạnh phải ROI:
    dương = xuống, âm = lên. Trả về (None, None) nếu không áp dụng."""
    y0 = _roi_top_split_y(poly, frac)
    if y0 is None:
        return None, None
    top = _clip_poly_half(poly, y0, keep_top=True)
    bot = _clip_poly_half(poly, y0, keep_top=False)
    if right_drop != 0.0:
        r2 = _right_edge_point(poly, y0, right_drop)
        if r2 is not None:
            _replace_right_y0_vertex(top, y0, r2)
            _replace_right_y0_vertex(bot, y0, r2)
    return top, bot
