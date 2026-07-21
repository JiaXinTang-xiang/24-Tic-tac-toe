#!/usr/bin/env python3
"""三子棋对弈系统 v3 - 点击棋盘选格子"""

import os, sys, json, time, cv2, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import customtkinter as ctk
from chess_detection import (detect_chessboard, draw_board_overlay,
                              load_calibration, init_undistort_maps)
from game_engine import GameEngine
from serial_client import SerialClient

_yolo_detect = None

DEFAULT_PARAMS = {
    "canny_low": 50, "canny_high": 150,
    "area_min": 7000, "area_max": 15000,
    "min_rectangularity": 0.65, "yolo_conf": 0.4,
}
PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
params = DEFAULT_PARAMS.copy()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

def cv2_to_ctk(cv_img, w=None, h=None):
    if cv_img is None: return None
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if w and h: pil = pil.resize((w, h), Image.LANCZOS)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)

def draw_virtual_board(pieces, highlight=-1, size=200):
    """画虚拟棋盘, highlight=高亮格子索引(-1=无)"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 220
    cs = size // 3
    for i in range(1, 3):
        cv2.line(img, (i*cs, 0), (i*cs, size), (50,50,50), 2)
        cv2.line(img, (0, i*cs), (size, i*cs), (50,50,50), 2)
    # 高亮格子
    if 0 <= highlight < 9:
        hr, hc = highlight // 3, highlight % 3
        cv2.rectangle(img, (hc*cs+2, hr*cs+2), ((hc+1)*cs-2, (hr+1)*cs-2), (0,200,255), 4)
    for idx in range(9):
        row, col = idx // 3, idx % 3
        cx, cy = col*cs + cs//2, row*cs + cs//2
        r = int(cs*0.35)
        if pieces[idx] == 1:
            cv2.circle(img, (cx,cy), r, (40,40,40), -1)
        elif pieces[idx] == 2:
            cv2.circle(img, (cx,cy), r, (230,230,230), -1)
            cv2.circle(img, (cx,cy), r, (50,50,50), 2)
    return img

def draw_crop_board(frame, board_box, pieces, highlight=-1, size=420):
    """透视矫正棋盘 + 棋子覆盖 + 高亮"""
    if board_box is None or frame is None: return None
    pts = board_box.astype(np.float32).reshape(4, 2)
    dst = np.array([[0,0],[size,0],[size,size],[0,size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(frame, M, (size, size))
    cs = size // 3
    for i in range(1, 3):
        cv2.line(crop, (i*cs,0), (i*cs,size), (255,0,0), 1)
        cv2.line(crop, (0,i*cs), (size,i*cs), (255,0,0), 1)
    if 0 <= highlight < 9:
        hr, hc = highlight // 3, highlight % 3
        cv2.rectangle(crop, (hc*cs+3, hr*cs+3), ((hc+1)*cs-3, (hr+1)*cs-3), (0,255,255), 4)
    for idx in range(9):
        row, col = idx // 3, idx % 3
        cx, cy = col*cs + cs//2, row*cs + cs//2
        r = int(cs*0.35)
        if pieces[idx] == 1:
            cv2.circle(crop, (cx,cy), r, (0,0,0), -1)
        elif pieces[idx] == 2:
            cv2.circle(crop, (cx,cy), r, (255,255,255), -1)
            cv2.circle(crop, (cx,cy), r, (0,0,0), 2)
    return crop

# =============================================================================
# 可点击棋盘 Frame
# =============================================================================
class ClickableBoard(ctk.CTkFrame):
    """可点击的棋盘组件: 显示棋盘图 + 点击回调"""
    def __init__(self, master, size=420, **kwargs):
        super().__init__(master, **kwargs)
        self.board_size = size
        self.on_cell_click = None  # 回调: func(cell_idx)

        self.label = ctk.CTkLabel(self, text="")
        self.label.pack(fill="both", expand=True)
        self.label.bind("<Button-1>", self._on_click)

    def _on_click(self, event):
        if self.on_cell_click is None: return
        cs = self.label.winfo_width() // 3
        if cs <= 0: return
        col = event.x // cs
        row = event.y // cs
        if 0 <= row < 3 and 0 <= col < 3:
            self.on_cell_click(row * 3 + col)

    def update_image(self, cv_img):
        w = self.label.winfo_width()
        h = self.label.winfo_height()
        if w < 50: w = self.board_size
        if h < 50: h = self.board_size
        ctk_img = cv2_to_ctk(cv_img, w, h)
        if ctk_img:
            self.label.configure(image=ctk_img, text="")


# =============================================================================
# 主应用
# =============================================================================
class TicTacToeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("三子棋对弈系统")
        self.geometry("1150x700")
        self.minsize(1100, 650)

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
        self.grid_selected = []   # 选中的目标格子
        self.highlight_cell = -1  # 鼠标悬停高亮

        self._init_camera()
        self._load_params()
        self._build_ui()
        self._start_video()

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
        except FileNotFoundError: pass

    def _save_params(self):
        with open(PARAMS_FILE, 'w') as f:
            json.dump(params, f, indent=2)

    # ==================================================================
    # UI
    # ==================================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=4)  # 棋盘区
        self.grid_columnconfigure(1, weight=3)  # 缩略图+预览
        self.grid_columnconfigure(2, weight=4)  # 控制面板
        self.grid_rowconfigure(0, weight=1)

        font_h = ctk.CTkFont(size=15, weight="bold")

        # ---- 左: 裁剪棋盘(大,可点击) ----
        self.crop_board = ClickableBoard(self, size=320)
        self.crop_board.grid(row=0, column=0, sticky="nsew", padx=(5,2), pady=5)
        self.crop_board.on_cell_click = self._on_board_click
        ctk.CTkLabel(self, text="点击棋盘格子选择目标", font=ctk.CTkFont(size=12),
                     text_color="gray").grid(row=1, column=0, sticky="n", pady=(0,3))

        # ---- 中: 虚拟棋盘 + 摄像头预览 ----
        mid = ctk.CTkFrame(self)
        mid.grid(row=0, column=1, sticky="nsew", padx=2, pady=5)
        mid.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(mid, text="虚拟棋盘 (点击选择)", font=font_h).pack(pady=(5,2))
        self.virtual_board = ClickableBoard(mid, size=200)
        self.virtual_board.pack(padx=5, pady=3)
        self.virtual_board.on_cell_click = self._on_board_click

        ctk.CTkLabel(mid, text="摄像头预览", font=font_h).pack(pady=(15,2))
        self.cam_preview = ctk.CTkLabel(mid, text="")
        self.cam_preview.pack(padx=5, pady=3, fill="both", expand=True)

        # ---- 右: 控制面板 ----
        right = ctk.CTkFrame(self)
        right.grid(row=0, column=2, sticky="nsew", padx=(2,5), pady=5, rowspan=2)
        right.grid_columnconfigure(0, weight=1)

        r = 0
        pad = {'padx':8, 'pady':2}

        # 题目
        ctk.CTkLabel(right, text="题目选择", font=font_h).grid(row=r, column=0, sticky="w", **pad); r += 1
        tf = ctk.CTkFrame(right, fg_color="transparent")
        tf.grid(row=r, column=0, sticky="ew", padx=5)
        for i, d in enumerate(["(1)单黑","(2)两黑两白","(3)旋转","(4)AI先手","(5)人先手","(6)作弊"], 1):
            ctk.CTkButton(tf, text=d, height=30, font=ctk.CTkFont(size=12),
                          command=lambda t=i: self._on_task(t)).grid(
                row=(i-1)//2, column=(i-1)%2, padx=2, pady=2, sticky="ew"); tf.grid_columnconfigure((i-1)%2, weight=1)
        r += 1

        # 执棋
        ctk.CTkLabel(right, text="执棋方", font=font_h).grid(row=r, column=0, sticky="w", **pad); r += 1
        sf = ctk.CTkFrame(right, fg_color="transparent")
        sf.grid(row=r, column=0, sticky="ew", padx=5)
        self.side_var = ctk.IntVar(value=1)
        ctk.CTkRadioButton(sf, text="系统执黑", variable=self.side_var, value=1,
                           font=ctk.CTkFont(size=14),
                           command=lambda: self._on_side(1)).pack(side="left", padx=15)
        ctk.CTkRadioButton(sf, text="系统执白", variable=self.side_var, value=2,
                           font=ctk.CTkFont(size=14),
                           command=lambda: self._on_side(2)).pack(side="left", padx=15)
        r += 1

        # 选中格子
        ctk.CTkLabel(right, text="已选目标", font=font_h).grid(row=r, column=0, sticky="w", **pad); r += 1
        self.target_label = ctk.CTkLabel(right, text="(未选择)", font=ctk.CTkFont(size=18, weight="bold"),
                                         fg_color="#1a2a3a", corner_radius=8, height=40)
        self.target_label.grid(row=r, column=0, sticky="ew", padx=8, pady=3)
        r += 1

        # 操作
        ctk.CTkLabel(right, text="操作", font=font_h).grid(row=r, column=0, sticky="w", **pad); r += 1
        bf = ctk.CTkFrame(right, fg_color="transparent")
        bf.grid(row=r, column=0, sticky="ew", padx=5)
        self.btn_confirm = ctk.CTkButton(bf, text="确认落子", height=45,
                                         font=ctk.CTkFont(size=17, weight="bold"),
                                         command=self._on_confirm)
        self.btn_confirm.pack(side="left", padx=3, fill="x", expand=True)
        self.btn_reset = ctk.CTkButton(bf, text="重检", width=65, height=45,
                                       fg_color="#555", command=self._on_reset)
        self.btn_reset.pack(side="left", padx=3)
        self.btn_clear = ctk.CTkButton(bf, text="清空", width=65, height=45,
                                       fg_color="#555", command=self._on_clear_target)
        self.btn_clear.pack(side="left", padx=3)
        r += 1

        # 状态
        ctk.CTkLabel(right, text="状态", font=font_h).grid(row=r, column=0, sticky="w", **pad); r += 1
        self.status_label = ctk.CTkLabel(right, text="就绪，选择题目开始",
                                         font=ctk.CTkFont(size=13),
                                         fg_color="#1a2a3a", corner_radius=8, height=45, padx=12)
        self.status_label.grid(row=r, column=0, sticky="ew", padx=8, pady=3); r += 1
        self.piece_label = ctk.CTkLabel(right, text="", font=ctk.CTkFont(size=12))
        self.piece_label.grid(row=r, column=0, sticky="w", padx=10, pady=2); r += 1

        # 调试面板
        self.debug_header = ctk.CTkButton(right, text="调试参数", height=28,
                                          fg_color="transparent", text_color="gray",
                                          command=self._toggle_debug)
        self.debug_header.grid(row=r, column=0, sticky="w", padx=8, pady=(10,2)); r += 1
        self.debug_frame = ctk.CTkFrame(right, fg_color="transparent")
        self.debug_visible = True
        self._build_debug(self.debug_frame)
        self.debug_frame.grid(row=r, column=0, sticky="ew", padx=5, pady=5)

    def _build_debug(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        rr = 0
        for key, label, lo, hi in [
            ("canny_low","Canny Low",5,250), ("canny_high","Canny High",10,255),
            ("area_min","Area Min",500,50000), ("area_max","Area Max",1000,200000),
            ("min_rectangularity","Rect度",30,95), ("yolo_conf","YOLO Conf",10,90),
        ]:
            ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=11), width=70).grid(row=rr, column=0, sticky="w")
            val = params[key]
            cur = int(val*100) if key in ("min_rectangularity","yolo_conf") else int(val)
            s = ctk.CTkSlider(parent, from_=lo, to=hi, height=14, number_of_steps=hi-lo,
                              command=lambda v,k=key: self._on_slider(k,v))
            s.set(cur); s.grid(row=rr, column=1, sticky="ew", padx=5, pady=2)
            vl = ctk.CTkLabel(parent, text=str(val), width=38, font=ctk.CTkFont(size=10))
            vl.grid(row=rr, column=2, sticky="e")
            setattr(self, f"_l_{key}", vl); setattr(self, f"_s_{key}", s)
            rr += 1
        bf2 = ctk.CTkFrame(parent, fg_color="transparent")
        bf2.grid(row=rr, column=0, columnspan=3, sticky="ew", pady=3)
        ctk.CTkButton(bf2, text="保存阈值", height=25, command=self._save_params).pack(side="left", padx=3)
        ctk.CTkButton(bf2, text="恢复默认", height=25, fg_color="#555", command=self._reset_params).pack(side="left", padx=3)

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

    def _on_board_click(self, idx):
        """点击棋盘格子: 切换选中"""
        if idx in self.grid_selected:
            self.grid_selected.remove(idx)
        else:
            self.grid_selected.append(idx)
        self.engine.set_target_grids([g+1 for g in self.grid_selected])
        sel_text = ", ".join(str(g+1) for g in sorted(self.grid_selected)) if self.grid_selected else "(未选择)"
        self.target_label.configure(text=f"格子: {sel_text}")

    def _on_clear_target(self):
        self.grid_selected.clear()
        self.target_label.configure(text="(未选择)")
        self.engine.set_target_grids([])

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
        self.squares_center = None; self.board_box = None
        self.pieces = [0]*9; self._on_clear_target()

    def _on_slider(self, key, value):
        global params
        if key in ("min_rectangularity","yolo_conf"):
            params[key] = round(value/100.0, 2)
        else:
            params[key] = int(value)
        lbl = getattr(self, f"_l_{key}", None)
        if lbl: lbl.configure(text=str(params[key]))

    def _reset_params(self):
        global params
        params.update(DEFAULT_PARAMS)
        for key, val in DEFAULT_PARAMS.items():
            s = getattr(self, f"_s_{key}", None)
            l = getattr(self, f"_l_{key}", None)
            if s: s.set(int(val*100) if key in ("min_rectangularity","yolo_conf") else val)
            if l: l.configure(text=str(val))

    def _update_status(self):
        self.status_label.configure(text=self.engine.get_status_text())
        b, w, e = self.engine.get_counts()
        sd = "黑" if self.engine.side==1 else "白"
        self.piece_label.configure(text=f"黑:{b} 白:{w} 空:{e} | 系统:{sd} | 题{self.engine.task}")

    # ==================================================================
    # 视频循环
    # ==================================================================
    def _start_video(self):
        self._update_video()

    def _update_video(self):
        if not self.running: return
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

                self.pieces = [0]*9; self.detections = []
                if self.squares_center:
                    try:
                        self.pieces, self.detections = self._yolo_detect(frame)
                    except Exception: pass

                if self.engine.status in ("playing","waiting_human","ai_thinking","done"):
                    self.engine.process_vision_result(self.pieces)

                # 裁剪棋盘 (大,可点击)
                hl = self.grid_selected[-1] if self.grid_selected else -1
                crop_img = draw_crop_board(frame, self.board_box, self.pieces, highlight=hl, size=320)
                self.crop_board.update_image(crop_img if crop_img is not None else frame)

                # 虚拟棋盘
                vb = draw_virtual_board(self.pieces, highlight=hl, size=200)
                self.virtual_board.update_image(vb)

                # 摄像头预览 (保持宽高比, 不拉伸)
                pw = self.cam_preview.winfo_width()
                ph = self.cam_preview.winfo_height()
                if pw > 50 and ph > 50:
                    # 等比缩放
                    fh, fw = frame.shape[:2]
                    scale = min((pw-4)/fw, (ph-4)/fh)
                    nw, nh = int(fw*scale), int(fh*scale)
                    preview = cv2.resize(frame, (nw, nh))
                    prv = draw_board_overlay(preview, self.squares_center,
                                             self.inner_radius, self.board_box,
                                             self.pieces, 0)
                    ctk_p = cv2_to_ctk(prv, nw, nh)
                    if ctk_p:
                        self.cam_preview.configure(image=ctk_p, text="")

                self._update_status()
        self.after(33, self._update_video)

    def _yolo_detect(self, frame):
        global _yolo_detect
        if _yolo_detect is None:
            from chess_detection import detect_pieces_yolo
            _yolo_detect = detect_pieces_yolo
        conf = params.get("yolo_conf", 0.4)
        return _yolo_detect(frame, self.squares_center, self.inner_radius, conf=conf)

    def on_close(self):
        self.running = False
        if self.cap: self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = TicTacToeApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
