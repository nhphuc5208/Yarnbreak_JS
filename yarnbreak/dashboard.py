"""OpenCV dashboard renderer.

Split out of main.py; code moved verbatim.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from yarnbreak.config import AppConfig
from yarnbreak.roi import _ROI_REF_HEIGHT_PX, _parse_roi, _roi_split_polys

logger = logging.getLogger("YarnBreak")


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


class DashboardRenderer:
    # ── Colour palette (BGR) ──────────────────────────────────────────────────
    _STYLE: Dict = {
        "dashboard_bg":       (14,  16,  20),
        "panel_bg":           (24,  27,  32),
        "video_bg":           ( 8,  10,  13),
        "header_bg":          (28,  32,  38),
        "footer_bg":          (22,  25,  30),
        "divider_color":      (45,  50,  58),
        "padding": 2,  "gap": 4,
        "header_height": 30,  "footer_height": 30,
        "border_thickness": 1,
        "font":               cv2.FONT_HERSHEY_SIMPLEX,
        "title_scale":        0.52,  "title_thickness": 2,
        "footer_label_scale": 0.43,  "footer_value_scale": 0.48,
        "footer_thickness":   1,
        "status_pill_scale":  0.42,  "status_pill_thickness": 1,
        "det_label_min":      0.48,  "det_label_max": 2.8,
        "det_label_mul":      1.08,  "det_label_t_plus": 0,
        "label_pad_x":        11,    "label_pad_y": 7,
        "warn_title_mul":     1.12,  "warn_title_min": 0.62, "warn_title_max": 2.8,
        "warn_sub_mul":       1.00,  "warn_sub_min": 0.42,   "warn_sub_max": 1.8,
        "warn_icon_mul":      1.20,  "warn_t_plus": 1,
        "original_accent":    (220, 220, 220),
        "detected_accent":    (120, 255, 180),
        "normal_color":       (120, 255, 180),
        "warning_color":      ( 40, 120, 255),
        "warning_red":        ( 40,  40, 255),
        "box_color":          ( 90, 210, 255),
        "mask_color":         ( 90, 190, 255),
        "roi_color":          ( 80, 230,  80),   # xanh lá – vùng dưới ROI (logic cũ)
        "roi_top_color":      ( 60, 180, 255),   # cam – vùng trên ROI (15% nhạy)
        "label_bg":           ( 30,  34,  40),
        "label_text":         (245, 245, 245),
        "badge_bg":           ( 30,  75, 210),
        "badge_alpha":        0.76,
        "badge_text":         (255, 255, 255),
        "status_text":        (230, 235, 240),
        "muted_text":         (165, 172, 182),
        "fps_color":          (120, 255, 180),
    }

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.S = self._STYLE
        self._vscale_cache:   Dict = {}
        self._tscale_cache:   Dict = {}
        self._layout_cache:   Dict = {}
        self._resize_cache:   Dict = {}
        self._dash_buf:       Optional[np.ndarray] = None
        self._src_label = "IPCAM" if cfg.source == "ipcam" else "VIDEO"
        # Đa giác ROI (vùng trục) dạng tỉ lệ 0–1, để vẽ viền lên video khi bật.
        self._roi = _parse_roi(cfg.roi) if cfg.roi_enabled else None
        # Hai đa giác vùng trên / dưới (đường chia có thể nghiêng) để vẽ 2 màu — phải
        # dựng GIỐNG HỆT detector để hình vẽ khớp đúng vùng phân loại.
        _drop = cfg.roi_top_right_drop_px / _ROI_REF_HEIGHT_PX
        self._roi_top_poly, self._roi_bot_poly = (
            _roi_split_polys(self._roi, cfg.roi_top_fraction, _drop)
            if self._roi is not None else (None, None))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vscale(self, h: int, w: int) -> Tuple:
        """Returns (font_scale, small_scale, thickness, box_thickness). Cached."""
        key = (h, w)
        if key not in self._vscale_cache:
            base = min(h, w)
            # Overlay được vẽ trên frame gốc (5MP → base~2048) rồi mới thu nhỏ vào
            # panel, nên trần phải đủ lớn để chữ/box không bị teo. Tỉ lệ ~tuyến tính
            # theo độ phân giải: 720p→~0.82, 1080p→~1.2, 5MP→~2.3.
            scl = self.cfg.overlay_scale
            fs = _clamp(base / 880.0,  0.48, 2.6) * scl
            ss = _clamp(base / 1050.0, 0.40, 2.0) * scl
            t  = int(_clamp(round(base / 390.0 * scl), 1, 7))
            bt = int(_clamp(round(base / 330.0 * scl), 2, 9))
            self._vscale_cache[key] = (fs, ss, t, bt)
        return self._vscale_cache[key]

    def _fit(self, text: str, desired: float, thick: int, max_w: int, min_s: float = 0.30) -> float:
        """Shrink font scale until text fits max_w. Cached."""
        key = (text, round(desired, 4), thick, max_w, round(min_s, 4))
        if key in self._tscale_cache:
            return self._tscale_cache[key]
        s = desired
        font = self.S["font"]
        while s > min_s:
            (tw, _), _ = cv2.getTextSize(text, font, s, thick)
            if tw <= max_w:
                break
            s -= 0.03
        r = max(s, min_s)
        self._tscale_cache[key] = r
        return r

    def _alpha_rect(self, img, x1, y1, x2, y2, color, alpha=0.6, border=None, bt=1):
        H, W = img.shape[:2]
        x1 = int(_clamp(x1, 0, W - 1)); y1 = int(_clamp(y1, 0, H - 1))
        x2 = int(_clamp(x2, 0, W - 1)); y2 = int(_clamp(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        layer = np.full(roi.shape, color, dtype=np.uint8)
        cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, roi)
        if border and bt > 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), border, bt, cv2.LINE_AA)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _layout(self, dw: int, dh: int) -> Dict:
        key = (dw, dh)
        if key in self._layout_cache:
            return self._layout_cache[key]
        S = self.S
        p, g, hh, fh = S["padding"], S["gap"], S["header_height"], S["footer_height"]
        uw = dw - 2 * p - g
        uh = dh - 2 * p
        lw = uw // 2
        rw = uw - lw
        ph = uh
        left = {
            "panel":  (p, p, lw, ph),
            "header": (p, p, lw, hh),
            "video":  (p, p + hh, lw, ph - hh - fh),
            "footer": (p, p + ph - fh, lw, fh),
        }
        rx = p + lw + g
        right = {
            "panel":  (rx, p, rw, ph),
            "header": (rx, p, rw, hh),
            "video":  (rx, p + hh, rw, ph - hh - fh),
            "footer": (rx, p + ph - fh, rw, fh),
        }
        result = {"left": left, "right": right}
        self._layout_cache[key] = result
        return result

    def _fit_video(self, canvas, img, rect, bg):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
        ih, iw = img.shape[:2]
        if ih == 0 or iw == 0 or w <= 0 or h <= 0:
            return
        key = (iw, ih, w, h)
        if key not in self._resize_cache:
            sc = min(w / iw, h / ih)
            nw = max(1, int(iw * sc))
            nh = max(1, int(ih * sc))
            self._resize_cache[key] = (nw, nh, (w - nw) // 2, (h - nh) // 2)
        nw, nh, dx, dy = self._resize_cache[key]
        rsz = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas[y + dy:y + dy + nh, x + dx:x + dx + nw] = rsz

    # ── Panel sub-components ──────────────────────────────────────────────────

    def _header(self, canvas, rect, title, accent, status=None, sc=None):
        S = self.S
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), S["header_bg"], -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + 2), accent, -1)
        ty = y + int(h * 0.70)
        ts = self._fit(title, S["title_scale"], S["title_thickness"], w - 160, 0.38)
        cv2.putText(canvas, title, (x + 10, ty), S["font"], ts, accent, S["title_thickness"], cv2.LINE_AA)
        if status:
            txt = status.upper()
            ps = self._fit(txt, S["status_pill_scale"], S["status_pill_thickness"], 120, 0.32)
            (tw, _), _ = cv2.getTextSize(txt, S["font"], ps, S["status_pill_thickness"])
            pw, ph2 = tw + 22, max(20, h - 8)
            px1 = x + w - pw - 8
            py1 = y + (h - ph2) // 2
            px2, py2 = px1 + pw, py1 + ph2
            c = sc or S["normal_color"]
            self._alpha_rect(canvas, px1, py1, px2, py2, c, 0.18, c, 1)
            cv2.putText(canvas, txt, (px1 + 10, py1 + ph2 - 6), S["font"], ps, c, S["status_pill_thickness"], cv2.LINE_AA)

    def _footer(self, canvas, rect, items):
        S = self.S
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), S["footer_bg"], -1)
        cx = x + 10
        ty = y + int(h * 0.72)
        ls, vs, t = S["footer_label_scale"], S["footer_value_scale"], S["footer_thickness"]
        for lbl, val, col in items:
            lt, vt = f"{lbl}:", str(val)
            cv2.putText(canvas, lt, (cx, ty), S["font"], ls, S["muted_text"], t, cv2.LINE_AA)
            (lw2, _), _ = cv2.getTextSize(lt, S["font"], ls, t)
            cv2.putText(canvas, vt, (cx + lw2 + 5, ty), S["font"], vs, col, t, cv2.LINE_AA)
            (vw2, _), _ = cv2.getTextSize(vt, S["font"], vs, t)
            cx += lw2 + vw2 + 24
            if cx > x + w - 80:
                break

    def _panel(self, canvas, img, lo, title, accent, fitems, status=None, sc=None):
        S = self.S
        px, py, pw, ph = lo["panel"]
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), S["panel_bg"], -1)
        self._header(canvas, lo["header"], title, accent, status, sc)
        self._fit_video(canvas, img, lo["video"], S["video_bg"])
        self._footer(canvas, lo["footer"], fitems)
        cv2.rectangle(canvas, (px, py), (px + pw - 1, py + ph - 1), S["divider_color"], S["border_thickness"], cv2.LINE_AA)

    # ── Detection drawing ─────────────────────────────────────────────────────

    def draw_roi(self, img: np.ndarray):
        """Vẽ viền vùng trục (ROI) lên frame để căn chỉnh. Toạ độ ROI là tỉ lệ
        0–1 nên nhân với kích thước frame hiện tại. Không làm gì nếu ROI tắt.
        Nếu bật vùng trên nhạy → tô 2 màu: cam = vùng trên (nhạy), xanh = vùng dưới."""
        if not self._roi:
            return
        h, w = img.shape[:2]
        _, _, _, bt = self._vscale(h, w)
        c_bot = self.S["roi_color"]
        c_top = self.S["roi_top_color"]

        def _to_px(poly_frac):
            if poly_frac is None or len(poly_frac) < 3:
                return None
            arr = np.array([(x * w, y * h) for x, y in poly_frac], dtype=np.int32)
            return arr.reshape((-1, 1, 2))

        # Không chia vùng → vẽ như cũ, một màu.
        if self._roi_top_poly is None:
            bands = [(self._roi, c_bot)]
        else:
            bands = [(self._roi_bot_poly, c_bot), (self._roi_top_poly, c_top)]

        overlay = img.copy()
        for poly_frac, c in bands:
            pts = _to_px(poly_frac)
            if pts is not None:
                cv2.fillPoly(overlay, [pts], c)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
        for poly_frac, c in bands:
            pts = _to_px(poly_frac)
            if pts is None:
                continue
            cv2.polylines(img, [pts], True, c, max(2, bt + 1), cv2.LINE_AA)
        # Chỉ chấm đỉnh của đa giác ROI GỐC (không chấm điểm giao do cắt band).
        outer = _to_px(self._roi)
        if outer is not None:
            for p in outer.reshape(-1, 2):
                cv2.circle(img, (int(p[0]), int(p[1])), max(4, bt + 3), c_bot, -1, cv2.LINE_AA)

    def draw_detections(self, img: np.ndarray, target: Optional[Dict]) -> int:
        if target is None:
            return 0
        S = self.S
        h, w = img.shape[:2]
        fs, ss, t, bt = self._vscale(h, w)
        conf = target["conf"]
        x1, y1, x2, y2 = target["box"]
        poly = target["poly"]
        box_c = S["box_color"]
        msk_c = S["mask_color"]

        # ── Segmentation mask ─────────────────────────────────────────────────
        if not self.cfg.fast_draw and self.cfg.enable_mask:
            if poly is not None and len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                arc = cv2.arcLength(pts, True)
                if arc > 0:
                    pts = cv2.approxPolyDP(pts, 0.004 * arc, True)
                bx, by, bw2, bh2 = cv2.boundingRect(pts)
                bx, by = max(0, bx), max(0, by)
                bx2, by2 = min(w, bx + bw2), min(h, by + bh2)
                if bx2 > bx and by2 > by:
                    roi = img[by:by2, bx:bx2]
                    roi_cp = roi.copy()
                    shifted = pts - np.array([[[bx, by]]], dtype=np.int32)
                    cv2.fillPoly(roi_cp, [shifted], msk_c)
                    cv2.addWeighted(roi_cp, self.cfg.mask_alpha, roi, 1 - self.cfg.mask_alpha, 0, roi)
                cv2.polylines(img, [pts], True, box_c, max(1, bt - 1), cv2.LINE_AA)
            else:
                bx, by = max(0, x1), max(0, y1)
                bx2, by2 = min(w, x2), min(h, y2)
                if bx2 > bx and by2 > by:
                    roi = img[by:by2, bx:bx2]
                    layer = np.full(roi.shape, msk_c, dtype=np.uint8)
                    cv2.addWeighted(layer, self.cfg.mask_alpha, roi, 1 - self.cfg.mask_alpha, 0, roi)

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(img, (x1, y1), (x2, y2), box_c, bt, cv2.LINE_AA)

        # ── Corner highlights ─────────────────────────────────────────────────
        cl = max(16, int(min(x2 - x1, y2 - y1) * 0.16))
        ct = bt + 1
        for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx1 = cl if px == x1 else -cl
            dy1 = cl if py == y1 else -cl
            cv2.line(img, (px, py), (px + dx1, py), box_c, ct, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy1), box_c, ct, cv2.LINE_AA)

        # ── Label ─────────────────────────────────────────────────────────────
        self._draw_label(img, f"{self.cfg.display_label} {conf:.2f}", x1, y1, box_c, h, w, t)
        return 1

    def _draw_label(self, img, text, x1, y1, color, H, W, t):
        S = self.S
        fs, _, _, _ = self._vscale(H, W)
        ds = _clamp(fs * S["det_label_mul"], S["det_label_min"], S["det_label_max"])
        t2 = max(1, t + S["det_label_t_plus"])
        px, py = S["label_pad_x"], S["label_pad_y"]
        fs2 = self._fit(text, ds, t2, int(W * 0.45), S["det_label_min"])
        (tw, th), bl = cv2.getTextSize(text, S["font"], fs2, t2)
        lw2, lh2 = tw + px * 2, th + py * 2 + bl
        lx1 = int(_clamp(x1, 8, W - lw2 - 8))
        ly1 = int(y1 - lh2 - 7)
        if ly1 < 8:
            ly1 = int(_clamp(y1 + 7, 8, H - lh2 - 8))
        lx2, ly2 = lx1 + lw2, ly1 + lh2
        self._alpha_rect(img, lx1, ly1, lx2, ly2, S["label_bg"], 0.88, color, 1)
        cv2.rectangle(img, (lx1, ly1), (lx1 + 5, ly2), color, -1)
        cv2.putText(img, text, (lx1 + px + 4, ly2 - py - bl), S["font"], fs2, S["label_text"], t2, cv2.LINE_AA)

    def draw_warning_badge(self, img: np.ndarray):
        S = self.S
        h, w = img.shape[:2]
        fs, ss, t, _ = self._vscale(h, w)
        badge_text, sub_text, icon = "YARN BREAK DETECTED", "CHECK YARN LINE", "!"
        bh = int(_clamp(h * 0.095, 54, 230))
        bw = int(_clamp(w * 0.42, 360, 1150))
        x1, y1, x2, y2 = 16, 16, 16 + bw, 16 + bh
        self._alpha_rect(img, x1, y1, x2, y2, S["badge_bg"], S["badge_alpha"], S["warning_color"], 1)
        isz = bh - 18
        ix1, iy1 = x1 + 10, y1 + 9
        ix2, iy2 = ix1 + isz, iy1 + isz
        self._alpha_rect(img, ix1, iy1, ix2, iy2, S["warning_red"], 0.82, (255, 255, 255), 1)
        iscale = _clamp(fs * S["warn_icon_mul"], 0.68, 3.2)
        wt = max(2, t + S["warn_t_plus"])
        cv2.putText(img, icon, (ix1 + int(isz * 0.36), iy2 - int(isz * 0.20)), S["font"], iscale, S["badge_text"], wt, cv2.LINE_AA)
        tx = ix2 + 14
        avw = x2 - tx - 14
        ts2 = self._fit(badge_text, _clamp(fs * S["warn_title_mul"], S["warn_title_min"], S["warn_title_max"]), wt, avw, S["warn_title_min"])
        ss2 = self._fit(sub_text,   _clamp(ss * S["warn_sub_mul"],   S["warn_sub_min"],   S["warn_sub_max"]),   1,  avw, S["warn_sub_min"])
        cv2.putText(img, badge_text, (tx, y1 + int(bh * 0.52)), S["font"], ts2, S["badge_text"], wt, cv2.LINE_AA)
        cv2.putText(img, sub_text,   (tx, y1 + int(bh * 0.82)), S["font"], ss2, (235, 238, 242), 1,  cv2.LINE_AA)

    def draw_warning_corners(self, img: np.ndarray):
        h, w = img.shape[:2]
        c = self.S["warning_color"]
        t  = int(_clamp(min(h, w) / 300, 2, 8))
        ln = int(_clamp(min(h, w) * 0.07, 28, 190))
        m  = int(_clamp(min(h, w) * 0.022, 10, 60))
        for px, py in [(m, m), (w - m, m), (m, h - m), (w - m, h - m)]:
            dx = ln if px == m else -ln
            dy = ln if py == m else -ln
            cv2.line(img, (px, py), (px + dx, py), c, t, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy), c, t, cv2.LINE_AA)

    # ── Full 2-panel dashboard ─────────────────────────────────────────────────

    def make_dashboard(
        self,
        original: np.ndarray,
        detected: np.ndarray,
        fps: float,
        detected_count: int,
        alert: bool,
        infer_ms: float,
    ) -> np.ndarray:
        S = self.S
        dw, dh = self.cfg.window_width, self.cfg.window_height

        if self.cfg.enable_dashboard_cache:
            if self._dash_buf is None or self._dash_buf.shape[:2] != (dh, dw):
                self._dash_buf = np.empty((dh, dw, 3), dtype=np.uint8)
            self._dash_buf[:] = S["dashboard_bg"]
            canvas = self._dash_buf
        else:
            canvas = np.full((dh, dw, 3), S["dashboard_bg"], dtype=np.uint8)

        lo = self._layout(dw, dh)
        status = "WARNING" if alert else "NORMAL"
        sc = S["warning_color"] if alert else S["normal_color"]

        det_rows = [
            ("STATUS", status, sc),
            ("FPS", f"{fps:.1f}", S["fps_color"]),
            ("CLS", detected_count, S["status_text"]),
            ("INF", f"{infer_ms:.0f}ms", S["muted_text"]),
        ]

        self._panel(
            canvas, original, lo["left"],
            "ORIGINAL FEED", S["original_accent"],
            [("MODE", "RAW", S["status_text"]), ("SRC", self._src_label, S["status_text"])],
            "LIVE", S["original_accent"],
        )
        self._panel(
            canvas, detected, lo["right"],
            "AI DETECTION", S["detected_accent"],
            det_rows,
            status, sc,
        )
        return canvas
