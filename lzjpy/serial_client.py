"""
串口通信模块 — 匹配 STM32 17字节协议

帧格式 (17 bytes):
  [0]     0xAA        帧头
  [1]     0x0F        长度
  [2]     CMD         命令字 (0xA1 = 放棋)
  [3-4]   src_x       uint16, 高字节在前
  [5-6]   src_y       uint16
  [7-8]   src_z       uint16
  [9-10]  dst_x       uint16
  [11-12] dst_y       uint16
  [13-14] dst_z       uint16
  [15]    xor8        校验 (byte[1]~byte[14] 逐字节异或)
  [16]    0x55        帧尾
"""

import struct
import time

# =====================================================================
# 坐标映射表 (来自 STM32 StepMotor.h)
# =====================================================================

# 九宫格坐标: 格子号(1-9) → (X, Y) 脉冲数
CHESS_GRID = {
    1: (392, 225),     # 左上
    2: (1175, 225),    # 中上
    3: (1958, 225),    # 右上
    4: (392, 675),     # 左中
    5: (1175, 675),    # 正中
    6: (1957, 675),    # 右中
    7: (392, 1125),    # 左下
    8: (1175, 1125),   # 中下
    9: (1958, 1125),   # 右下
}

# 棋子放置区坐标: 取棋位置
# 左侧 (黑棋区): Chessplace[0]~[4]
# 右侧 (白棋区): Chessplace[5]~[9]
# 当前使用第一个有效位作为黑/白棋取棋坐标
BLACK_PIECE_ZONE = (220, 350)    # 黑棋取棋位置 (Chessplace[0])
WHITE_PIECE_ZONE = (2030, 123)   # 白棋取棋位置 (Chessplace[5])

# Z轴: 棋子区和棋盘都在同一平面, src_z/dst_z 始终为 0
Z_DEFAULT = 0

# =====================================================================
# 协议常量
# =====================================================================
FRAME_HEADER = 0xAA
FRAME_FOOTER = 0x55
FRAME_LENGTH = 0x0F
CMD_PLACE_PIECE = 0xA1
FRAME_SIZE = 17


def _xor_checksum(data: bytes) -> int:
    """计算 XOR 校验: byte[0]~byte[12] 逐字节异或 (对应帧中 byte[1]~byte[14])"""
    result = 0
    for b in data:
        result ^= b
    return result & 0xFF


def build_frame(cmd: int, src_x: int, src_y: int, src_z: int,
                dst_x: int, dst_y: int, dst_z: int) -> bytes:
    """
    构建 17 字节数据帧.

    参数:
        cmd:   命令字 (0xA1 = 放棋)
        src_x, src_y, src_z: 取棋位置坐标 (uint16)
        dst_x, dst_y, dst_z: 放棋位置坐标 (uint16)
    """
    # payload: cmd(1B) + 6个坐标(12B) = 13 字节 → 对应帧的 byte[2]~byte[14]
    payload = struct.pack('>BHHHHHH',
                          cmd & 0xFF,
                          src_x & 0xFFFF, src_y & 0xFFFF, src_z & 0xFFFF,
                          dst_x & 0xFFFF, dst_y & 0xFFFF, dst_z & 0xFFFF)

    # 校验段: 长度(1B) + payload(13B) = byte[1]~byte[14]
    check_data = struct.pack('>B', FRAME_LENGTH) + payload
    checksum = _xor_checksum(check_data)

    frame = struct.pack('>BB', FRAME_HEADER, FRAME_LENGTH) + payload \
            + struct.pack('>BB', checksum, FRAME_FOOTER)

    assert len(frame) == FRAME_SIZE, f"帧长度错误: {len(frame)}"
    return frame


# =====================================================================
# SerialClient
# =====================================================================
class SerialClient:
    """串口通信客户端"""

    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.enabled = False

    def connect(self, port=None, baudrate=None):
        """连接串口"""
        if port:
            self.port = port
        if baudrate:
            self.baudrate = baudrate

        if not self.port:
            print("[SERIAL] 未配置串口")
            return False

        try:
            import serial
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.enabled = True
            print(f"[SERIAL] {self.port} @ {self.baudrate} 已连接")
            return True
        except Exception as e:
            print(f"[SERIAL] 连接失败: {e}")
            self.enabled = False
            return False

    def disconnect(self):
        """断开串口"""
        if self.serial:
            self.serial.close()
            self.serial = None
        self.enabled = False

    def send_piece_command(self, from_pos, to_pos):
        """
        发送落子指令到机械臂.

        参数:
            from_pos: 棋子放置区编号 (0=黑棋区, 1=白棋区)
            to_pos:   目标格子编号 (1-9)
        """
        # 坐标映射
        if from_pos == 0:
            src_x, src_y = BLACK_PIECE_ZONE
        else:
            src_x, src_y = WHITE_PIECE_ZONE

        dst_x, dst_y = CHESS_GRID.get(to_pos, (0, 0))

        frame = build_frame(cmd=CMD_PLACE_PIECE,
                            src_x=src_x, src_y=src_y, src_z=Z_DEFAULT,
                            dst_x=dst_x, dst_y=dst_y, dst_z=Z_DEFAULT)

        if not self.enabled or self.serial is None:
            # 串口未连接: 打印模拟信息
            from_zone = "黑棋区" if from_pos == 0 else "白棋区"
            print(f"[SERIAL] 模拟发送: {from_zone}({src_x},{src_y}) → 格子{to_pos}({dst_x},{dst_y})")
            print(f"        帧数据: {frame.hex(' ').upper()}")
            return True

        try:
            self.serial.write(frame)
            from_zone = "黑棋区" if from_pos == 0 else "白棋区"
            print(f"[SERIAL] 已发送: {from_zone}({src_x},{src_y}) → 格子{to_pos}({dst_x},{dst_y})")
            return True
        except Exception as e:
            print(f"[SERIAL] 发送失败: {e}")
            return False

    def read_line(self, timeout=0.1):
        """读取一行串口数据 (用于接收按钮事件等)"""
        if not self.enabled or self.serial is None:
            return None
        try:
            self.serial.timeout = timeout
            return self.serial.readline()
        except Exception:
            return None
