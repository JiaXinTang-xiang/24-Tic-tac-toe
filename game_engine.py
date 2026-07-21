"""
游戏引擎 — 三子棋对弈核心逻辑
管理棋盘状态、流程控制、作弊检测、AI决策
"""

from ai import find_best_move, check_win


class GameEngine:
    def __init__(self, serial_client=None):
        self.board = [[0] * 3 for _ in range(3)]      # 当前棋盘
        self.prev_board = [[0] * 3 for _ in range(3)]  # 上一步(作弊检测)
        self.side = 1          # 1=系统执黑, 2=系统执白
        self.task = 4          # 当前题目 1-6
        self.turn = "system"   # 当前轮到谁: system/human
        self.winner = -1       # -1=未结束, 0=平局, 1=黑赢, 2=白赢
        self.target_grids = [] # 题目(1)(2)(3)的目标格子列表
        self.step_count = 0    # 已下步数
        self.status = "idle"   # idle/playing/waiting_human/ai_thinking/done
        self.serial = serial_client

    def reset(self):
        self.board = [[0] * 3 for _ in range(3)]
        self.prev_board = [[0] * 3 for _ in range(3)]
        self.winner = -1
        self.turn = "system"
        self.step_count = 0
        self.status = "idle"

    # ---- 题目流程 ----

    def start_task(self, task_num, side=1):
        """启动指定题目"""
        self.reset()
        self.task = task_num
        self.side = side
        self.status = "playing"

        if task_num in (1, 2, 3):
            self.turn = "system"
        elif task_num == 4:
            self.turn = "system"  # 系统先手黑棋
        elif task_num == 5:
            self.turn = "human"   # 人先手黑棋
        elif task_num == 6:
            self.turn = "human"

        return self.get_status_text()

    def set_target_grids(self, grids):
        """设置题目(1)(2)(3)的目标格子"""
        self.target_grids = grids

    def process_vision_result(self, pieces_1d):
        """
        视觉检测结果 → 更新棋盘.
        pieces_1d: [0,1,2,...] x9 (0=空,1=黑,2=白)
        """
        self.prev_board = [row[:] for row in self.board]
        for i in range(3):
            for j in range(3):
                self.board[i][j] = pieces_1d[i * 3 + j]

    def human_confirmed(self):
        """人按了确认按钮 → 处理当前回合"""
        if self.status == "done":
            return self.get_status_text()

        # 检查作弊 (题目6)
        if self.task == 6:
            moved = self.detect_cheat()
            if moved:
                self.status = "cheat_detected"
                return self.get_status_text()

        # 检查胜负
        self.winner = check_win(self.board)
        if self.winner != -1:
            self.status = "done"
            return self.get_status_text()

        # 轮到AI
        self.turn = "system"
        self.status = "ai_thinking"
        return self.get_status_text()

    def ai_make_move(self):
        """AI计算落子并返回位置"""
        if self.status != "ai_thinking":
            return None

        # 题目(1)(2)(3): 按预设目标格子走
        if self.task in (1, 2, 3):
            if self.step_count < len(self.target_grids):
                pos = self.target_grids[self.step_count]
                row, col = (pos - 1) // 3, (pos - 1) % 3
            else:
                return None
        else:
            # 对弈: Minimax
            move = find_best_move(self.board, self.side)
            if move is None:
                self.status = "done"
                return None
            row, col = move

        # 落子
        self.board[row][col] = self.side
        self.step_count += 1

        # 发送串口指令
        piece_zone = 0 if self.side == 1 else 1
        grid_num = row * 3 + col + 1
        if self.serial:
            self.serial.send_piece_command(piece_zone, grid_num)
        else:
            print(f"[ENGINE] AI落子: 格子{grid_num} (side={self.side})")

        # 检查胜负
        self.winner = check_win(self.board)
        if self.winner != -1:
            self.status = "done"
        else:
            self.turn = "human"
            self.status = "waiting_human"

        return grid_num

    def detect_cheat(self):
        """
        作弊检测 (题目6).
        返回被移动的棋子信息 [{from, to}] 或 None
        """
        moved = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] != self.prev_board[i][j]:
                    old = self.prev_board[i][j]
                    new = self.board[i][j]
                    # 系统棋子被移走 → 作弊
                    if old == self.side and new != self.side:
                        moved.append({"row": i, "col": j, "old": old, "new": new})
        return moved if moved else None

    def restore_cheated_piece(self, row, col):
        """将被移动的棋子放回原位"""
        self.board[row][col] = self.side
        grid_num = row * 3 + col + 1
        if self.serial:
            piece_zone = 0 if self.side == 1 else 1
            self.serial.send_piece_command(piece_zone, grid_num)
        return grid_num

    # ---- 状态查询 ----

    def get_board_1d(self):
        return [self.board[i][j] for i in range(3) for j in range(3)]

    def get_status_text(self):
        if self.status == "idle":
            return "就绪，选择题目开始"
        if self.status == "done":
            if self.winner == 0:
                return "平局!"
            win_name = "系统" if self.winner == self.side else "对手"
            return f"{win_name} 获胜!"
        if self.status == "cheat_detected":
            return "检测到作弊! 正在恢复棋子..."
        if self.status == "ai_thinking":
            return "AI 思考中..."
        if self.status == "waiting_human":
            return "等待对手落子..."
        if self.status == "playing":
            return f"题目{self.task} 进行中..."
        return "..."

    def get_counts(self):
        b = sum(cell == 1 for row in self.board for cell in row)
        w = sum(cell == 2 for row in self.board for cell in row)
        return b, w, 9 - b - w
