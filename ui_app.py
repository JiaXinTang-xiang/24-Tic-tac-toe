#!/usr/bin/env python3
"""
三子棋对弈系统 — CustomTkinter UI v2
- 左侧上半: 摄像头主画面
- 左侧下半: 裁剪棋盘缩略图（透视矫正+棋子叠加）
- 右侧: 大控制面板 + 可折叠调试参数
- 鼠标全操控
"""

import os, sys, json, time, cv2, numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customtkinter as ctk
from chess_detection import (detect_chessboard, draw_board_overlay,
                              load_calibration, init_undistort_maps)
from game_engine import GameEngine
from serial_client import SerialClient

_yolo_detect = None

# =============================================================================
# 参数
# =============================================================================
DEFAULT_PARAMS = {
    "canny_low": 50, "canny_high": 150,
    "area_min": 7000, "area_max": 15000,
    "min_rectangularity": 0.65, "yolo_conf": 0.4,
}
PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
params = DEFAULT_PARAMS.copy()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# =============================================================================
# 辅助
# =============================================================================
def cv2_to_ctk(cv_img, w=None, h=None):
    if cv_img is None:
        return None
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if w and h:
        pil = pil.resize((w, h), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)


def draw_virtual_board(pieces, size=240):
    """画虚拟棋盘: 白色背景 + 黑线 + 棋子"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 220
    cs = size // 3
    # 网格线
    for i in range(1, 3):
        cv2.line(img, (i * cs, 0), (i * cs, size), (50, 50, 50), 2)
        cv2.line(img, (0, i * cs), (size, i * cs), (50, 50, 50), 2)
    # 棋子
    for idx in range(9):
        row, col = idx // 3, idx % 3
        cx, cy = col * cs + cs // 2, row * cs + cs // 2
        r = int(cs * 0.35)
        if pieces[idx] == 1:
            cv2.circle(img, (cx, cy), r, (40, 40, 40), -1)
        elif pieces[idx] == 2:
            cv2.circle(img, (cx, cy), r, (230, 230, 230), -1)
            cv2.circle(img, (cx, cy), r, (50, 50, 50), 2)
    return img


# =============================================================================
# 主应用
# =============================================================================
class TicTacToeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("三子棋对弈系统")
        self.geometry("1400x850")
        self.minsize(1200, 700)

        # 状态
        self.engine = GameEngine(SerialClient())
        self.cap = None
        self.overlay = None
        self.running = True
        self.use_undistort = False
        self.map1 = self.map2 = None
        self.squares_center = None
        self.inner_radius = 30
        self.board_box = None
        self.pieces = [0] * 9
        self.detections = []

        self._init_camera()
        self._load_params()
        self._build_ui()
        self._start_video()

    # ==================================================================
    # 相机
    # ==================================================================
    def _init_camera(self):
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "camera_calibration.yaml")
        try:
            cm, dc, Pm = load_calibration(yaml_path)
            self.map1, self.map2 = init_undistort_maps(cm, dc)
            self.use_undistort = True
        except Exception:
            self.use_undistort = False
        for cid in [2, 0, 8]:
            cap = cv2.VideoCapture(cid)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap = cap
                return

    def _load_params(self):
        global params
        try:
            with open(PARAMS_FILE) as f:
                params.update(json.load(f))
        except FileNotFoundError:
            pass

    def _save_params(self):
        with open(PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2)

    # ==================================================================
    # UI 构建
    # ==================================================================
    def _build_ui(self):
        # 三列: 视频 | 棋盘缩略 | 控制面板
        self.grid_columnconfigure(0, weight=4)  # 视频
        self.grid_columnconfigure(1, weight=3)  # 棋盘缩略
        self.grid_columnconfigure(2, weight=3)  # 控制面板
        self.grid_rowconfigure(0, weight=1)

        # ---- 左: 视频 ----
        vf = ctk.CTkFrame(self, fg_color="black")
        vf.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        vf.grid_rowconfigure(0, weight=1)
        vf.grid_columnconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(vf, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # ---- 中: 棋盘缩略图 ----
        mid = ctk.CTkFrame(self)
        mid.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        mid.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(mid, text="虚拟棋盘", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        self.virtual_board_label = ctk.CTkLabel(mid, text="")
        self.virtual_board_label.grid(row=1, column=0, sticky="n", padx=5, pady=5)

        ctk.CTkLabel(mid, text="裁剪棋盘 (透视矫正)", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=2, column=0, sticky="w", padx=8, pady=(15, 2))
        self.crop_board_label = ctk.CTkLabel(mid, text="")
        self.crop_board_label.grid(row=3, column=0, sticky="n", padx=5, pady=5)

        # ---- 右: 控制面板 (不滚动, 更大) ----
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=2, sticky="nsew", padx=3, pady=3)
        right.grid_columnconfigure(0, weight=1)

        r = self._build_controls(right, 0)
        # 调试面板填充剩余空间
        right.grid_rowconfigure(r + 1, weight=1)

    def _build_controls(self, parent, start_row):
        """构建右侧所有控件, 返回最后行号"""
        pad = {'padx': 8, 'pady': 2}
        r = start_row
        font_h = ctk.CTkFont(size=15, weight="bold")
        font_n = ctk.CTkFont(size=14)

        # -- 题目 --
        ctk.CTkLabel(parent, text="题目选择", font=font_h).grid(row=r, column=0, sticky="w", **pad)
        r += 1
        tf = ctk.CTkFrame(parent, fg_color="transparent")
        tf.grid(row=r, column=0, sticky="ew", padx=5)
        for i, d in enumerate(["(1)单黑", "(2)两黑两白", "(3)旋转棋盘",
                                "(4)AI先手黑", "(5)人先手黑", "(6)作弊检测"], 1):
            ctk.CTkButton(tf, text=d, height=32, font=ctk.CTkFont(size=13),
                          command=lambda t=i: self._on_task(t)).grid(
                row=(i-1)//2, column=(i-1)%2, padx=3, pady=3, sticky="ew")
        r += 1

        # -- 执棋 --
        ctk.CTkLabel(parent, text="执棋方", font=font_h).grid(row=r, column=0, sticky="w", **pad)
        r += 1
        sf = ctk.CTkFrame(parent, fg_color="transparent")
        sf.grid(row=r, column=0, sticky="ew", padx=5)
        self.side_var = ctk.IntVar(value=1)
        ctk.CTkRadioButton(sf, text="系统 执黑棋", variable=self.side_var, value=1,
                           font=font_n, command=lambda: self._on_side(1)).pack(side="left", padx=10)
        ctk.CTkRadioButton(sf, text="系统 执白棋", variable=self.side_var, value=2,
                           font=font_n, command=lambda: self._on_side(2)).pack(side="left", padx=10)
        r += 1

        # -- 目标格子 --
        ctk.CTkLabel(parent, text="目标格子 (题目1-3)", font=font_h).grid(row=r, column=0, sticky="w", **pad)
        r += 1
        gf = ctk.CTkFrame(parent, fg_color="transparent")
        gf.grid(row=r, column=0, sticky="ew", padx=5)
        self.grid_btns = []
        self.grid_selected = []
        for i in range(9):
            btn = ctk.CTkButton(gf, text=str(i+1), width=55, height=38,
                                fg_color="#444", font=ctk.CTkFont(size=15),
                                command=lambda n=i+1: self._on_grid_select(n))
            btn.grid(row=i//3, column=i%3, padx=4, pady=4)
            self.grid_btns.append(btn)
        r += 1

        # -- 操作按钮 --
        ctk.CTkLabel(parent, text="操作", font=font_h).grid(row=r, column=0, sticky="w", **pad)
        r += 1
        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.grid(row=r, column=0, sticky="ew", padx=5)
        self.btn_confirm = ctk.CTkButton(bf, text="确认落子", height=40,
                                         font=ctk.CTkFont(size=16, weight="bold"),
                                         command=self._on_confirm)
        self.btn_confirm.pack(side="left", padx=5, fill="x", expand=True)
        self.btn_reset = ctk.CTkButton(bf, text="重检", width=70, height=40,
                                       fg_color="#555", command=self._on_reset)
        self.btn_reset.pack(side="left", padx=5)
        self.btn_save = ctk.CTkButton(bf, text="截图", width=70, height=40,
                                      fg_color="#555", command=self._on_screenshot)
        self.btn_save.pack(side="left", padx=5)
        r += 1

        # -- 状态 --
        ctk.CTkLabel(parent, text="状态", font=font_h).grid(row=r, column=0, sticky="w", **pad)
        r += 1
        self.status_label = ctk.CTkLabel(parent, text="就绪，选择题目开始",
                                         font=ctk.CTkFont(size=14),
                                         fg_color="#1a2a3a", corner_radius=8,
                                         height=45, padx=12)
        self.status_label.grid(row=r, column=0, sticky="ew", padx=8, pady=3)
        r += 1
        self.piece_label = ctk.CTkLabel(parent, text="", font=ctk.CTkFont(size=13))
        self.piece_label.grid(row=r, column=0, sticky="w", padx=10, pady=2)
        r += 1

        # -- 调试面板 (可折叠) --
        self.debug_header = ctk.CTkButton(parent, text="调试参数", height=30,
                                          fg_color="transparent", text_color="gray",
                                          command=self._toggle_debug)
        self.debug_header.grid(row=r, column=0, sticky="w", padx=8, pady=(15, 2))
        r += 1

        self.debug_frame = ctk.CTkFrame(parent, fg_color="transparent")
        self.debug_visible = True
        self._build_debug_panel(self.debug_frame)
        self.debug_frame.grid(row=r, column=0, sticky="ew", padx=5, pady=5)
        r += 1

        return r

    def _build_debug_panel(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        rr = 0   
        slider_configs = [
            ("canny_low", "Canny Low", 5, 250, 1),
            ("canny_high", "Canny High", 10, 255, 1),
            ("area_min", "Area Min", 500, 50000, 100),
            ("area_max", "Area Max", 1000, 200000, 1000),
            ("min_rectangularity", "矩形度", 30, 95, 1),
            ("yolo_conf", "YOLO置信度", 10, 90, 1),
        ]
        for key, label, lo, hi, _ in slider_configs:
            ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12), width=85).grid(
                row=rr, column=0, sticky="w", padx=3)
            val = params[key]
            if key in ("min_rectangularity", "yolo_conf"):
                cur = int(val * 100)
            else:
                cur = int(val)
            slider = ctk.CTkSlider(parent, from_=lo, to=hi, height=16,
                                   number_of_steps=hi - lo,
                                   command=lambda v, k=key: self._on_slider(k, v))
            slider.set(cur)
            slider.grid(row=rr, column=1, sticky="ew", padx=5, pady=2)
            vl = ctk.CTkLabel(parent, text=str(val), width=42, font=ctk.CTkFont(size=11))
            vl.grid(row=rr, column=2, sticky="e", padx=3)
            setattr(self, f"_lbl_{key}", vl)
            setattr(self, f"_sld_{key}", slider)
            rr += 1

        bf2 = ctk.CTkFrame(parent, fg_color="transparent")
        bf2.grid(row=rr, column=0, columnspan=3, sticky="ew", pady=5)
        ctk.CTkButton(bf2, text="保存阈值", height=28, command=self._save_params).pack(side="left", padx=5)
        ctk.CTkButton(bf2, text="恢复默认", height=28, fg_color="#555",
                      command=self._reset_params).pack(side="left", padx=5)

    def _toggle_debug(self):
        if self.debug_visible:
            self.debug_frame.grid_forget()
            self.debug_header.configure(text="调试参数")
        else:
            r = self.debug_header.grid_info()["row"] + 1
            self.debug_frame.grid(row=r, column=0, sticky="ew", padx=5, pady=5)
            self.debug_header.configure(text="调试参数")
        self.debug_visible = not self.debug_visible

    # ==================================================================
    # 事件
    # ==================================================================
    def _on_task(self, t):
        self.engine.start_task(t, self.side_var.get())
        self._update_status()

    def _on_side(self, s):
        self.engine.side = s

    def _on_grid_select(self, n):
        if n in self.grid_selected:
            self.grid_selected.remove(n)
            self.grid_btns[n-1].configure(fg_color="#444")
        else:
            self.grid_selected.append(n)
            self.grid_btns[n-1].configure(fg_color="#1f6aa5")
        self.engine.set_target_grids(self.grid_selected)

    def _on_confirm(self):
        self.engine.process_vision_result(self.pieces)
        self._update_status()
        if self.engine.status == "ai_thinking":
            self.engine.ai_make_move()
            self._update_status()
        if self.engine.status == "cheat_detected":
            for m in (self.engine.detect_cheat() or []):
                self.engine.restore_cheated_piece(m["row"], m["col"])

    def _on_reset(self):
        self.squares_center = None
        self.board_box = None
        self.pieces = [0] * 9
        self.grid_selected.clear()
        for b in self.grid_btns:
            b.configure(fg_color="#444")

    def _on_screenshot(self):
        if self.overlay is not None:
            cv2.imwrite(f"shot_{int(time.time())}.jpg", self.overlay)

    def _on_slider(self, key, value):
        global params
        if key in ("min_rectangularity", "yolo_conf"):
            params[key] = round(value / 100.0, 2)
        else:
            params[key] = int(value)
        lbl = getattr(self, f"_lbl_{key}", None)
        if lbl:
            lbl.configure(text=str(params[key]))

    def _reset_params(self):
        global params
        params.update(DEFAULT_PARAMS)
        for key, val in DEFAULT_PARAMS.items():
            s = getattr(self, f"_sld_{key}", None)
            l = getattr(self, f"_lbl_{key}", None)
            if s:
                s.set(int(val * 100) if key in ("min_rectangularity", "yolo_conf") else val)
            if l:
                l.configure(text=str(val))

    def _update_status(self):
        self.status_label.configure(text=self.engine.get_status_text())
        b, w, e = self.engine.get_counts()
        self.piece_label.configure(
            text=f"棋盘 黑:{b} 白:{w} 空:{e} | 系统:{{'黑' if self.engine.side==1 else '白'}} | 题{self.engine.task}".replace('{{', '{').replace('}}', '}'))

    # ==================================================================
    # 视频循环
    # ==================================================================
    def _start_video(self):
        self._update_video()

    def _update_video(self):
        if not self.running:
            return
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                if self.use_undistort:
                    frame = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

                cl, ch = int(params["canny_low"]), int(params["canny_high"])
                amin, amax = int(params["area_min"]), int(params["area_max"])
                result = detect_chessboard(frame, canny_low=cl, canny_high=ch,
                                           area_min=amin, area_max=amax)
                self.squares_center, self.inner_radius, self.board_box, _, _ = result

                self.pieces = [0] * 9
                self.detections = []
                if self.squares_center:
                    try:
                        self.pieces, self.detections = self._yolo_detect(frame)
                    except Exception:
                        pass

                if self.engine.status in ("playing", "waiting_human", "ai_thinking", "done"):
                    self.engine.process_vision_result(self.pieces)

                # 主画面
                self.overlay = draw_board_overlay(frame, self.squares_center,
                                                  self.inner_radius, self.board_box,
                                                  self.pieces, 0)
                for cx, cy, pt, xyxy in self.detections:
                    c = (30, 30, 30) if pt == 1 else (240, 240, 240)
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(self.overlay, (x1, y1), (x2, y2), c, 2)

                # 更新视频
                vw = self.video_label.winfo_width()
                vh = self.video_label.winfo_height()
                if vw > 50 and vh > 50:
                    ctk_img = cv2_to_ctk(self.overlay, vw - 6, vh - 6)
                    if ctk_img:
                        self.video_label.configure(image=ctk_img, text="")

                # 更新虚拟棋盘
                vb = self.virtual_board_label.winfo_width()
                if vb < 50:
                    vb = 250
                v_img = draw_virtual_board(self.pieces, size=vb)
                ctk_v = cv2_to_ctk(v_img, vb, vb)
                if ctk_v:
                    self.virtual_board_label.configure(image=ctk_v, text="")

                # 更新裁剪棋盘 (透视矫正)
                cb = self.crop_board_label.winfo_width()
                if cb < 50:
                    cb = 250
                if self.board_box is not None:
                    crop = self._crop_board(frame, size=cb)
                    if crop is not None:
                        ctk_c = cv2_to_ctk(crop, cb, cb)
                        if ctk_c:
                            self.crop_board_label.configure(image=ctk_c, text="")

                self._update_status()

        self.after(33, self._update_video)

    def _crop_board(self, frame, size):
        if self.board_box is None:
            return None
        pts = self.board_box.astype(np.float32).reshape(4, 2)
        dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts, dst)
        crop = cv2.warpPerspective(frame, M, (size, size))
        # 叠加棋子检测
        cs = size // 3
        for idx in range(9):
            row, col = idx // 3, idx % 3
            cx, cy = col * cs + cs // 2, row * cs + cs // 2
            r = int(cs * 0.3)
            if self.pieces[idx] == 1:
                cv2.circle(crop, (cx, cy), r, (0, 0, 0), -1)
            elif self.pieces[idx] == 2:
                cv2.circle(crop, (cx, cy), r, (255, 255, 255), -1)
                cv2.circle(crop, (cx, cy), r, (0, 0, 0), 2)
        return crop

    def _yolo_detect(self, frame):
        global _yolo_detect
        if _yolo_detect is None:
            from chess_detection import detect_pieces_yolo
            _yolo_detect = detect_pieces_yolo
        conf = params.get("yolo_conf", 0.4)
        return _yolo_detect(frame, self.squares_center, self.inner_radius, conf=conf)

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = TicTacToeApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
