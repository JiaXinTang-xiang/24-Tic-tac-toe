"""
串口通信模块 (参考 视觉代码带注释.py pack_frame 协议)
机械臂控制 + 蓝牙数据收发
"""

import struct
import time

# 协议常量
FRAME_HEADER = 0xA5


def pack_frame(cmd_id, flags, floats):
    """
    打包数据帧.
    帧结构: 帧头(0xA5) + 长度(1B) + cmd_id(2B) + flags(2B) + floats(N×4B)
    """
    n = len(floats)
    length = 2 + 2 + 4 * n
    buf = bytearray([FRAME_HEADER, length & 0xFF])
    buf += struct.pack('<HH', cmd_id, flags)
    for f in floats:
        buf += struct.pack('<f', float(f))
    return bytes(buf)


class SerialClient:
    """串口通信客户端 — 空壳，接上机械臂后填实际串口"""

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
        if self.serial:
            self.serial.close()
            self.serial = None
        self.enabled = False

    def send_piece_command(self, from_pos, to_pos):
        """
        发送落子指令到机械臂.
        from_pos: 棋子放置区编号 (0=黑棋区, 1=白棋区)
        to_pos:   目标格子编号 (1-9)
        """
        if not self.enabled:
            print(f"[SERIAL] 模拟发送: 从棋子区{from_pos} 放到格子{to_pos}")
            return True

        data = [float(from_pos), float(to_pos), 0.0]
        frame = pack_frame(cmd_id=0x0200, flags=0x0000, floats=data)
        try:
            self.serial.write(frame)
            print(f"[SERIAL] 已发送: 落子 {from_pos}→{to_pos}")
            return True
        except Exception as e:
            print(f"[SERIAL] 发送失败: {e}")
            return False

    def send_status(self, status_code):
        """发送状态码 (0=就绪, 1=下棋中, 2=完成, 3=错误)"""
        if not self.enabled:
            return
        frame = pack_frame(cmd_id=0x0300, flags=0x0000, floats=[float(status_code)])
        try:
            self.serial.write(frame)
        except Exception:
            pass
