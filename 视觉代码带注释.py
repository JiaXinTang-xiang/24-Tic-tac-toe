# -*- coding: utf-8 -*-
"""
电子设计竞赛激光追踪小车视觉系统 - 海康工业相机版本
Laser Tracking Car Vision System for Electronic Design Competition

作者 (Authors): 江南大学 郭兰鑫, 王昊 
开源协议 (License): GNU General Public License v3.0 (GPL-3.0)
项目描述 (Project Description): 
    专为电子设计竞赛开发的激光追踪小车视觉导航系统，基于海康威视工业相机
    实现目标检测、透视校正、激光点追踪和距离测量功能

系统架构 (System Architecture):
    ┌─────────────────────────────────────────────────────────────────┐
    │                    电赛激光追踪小车视觉系统                      │
    ├─────────────────────────────────────────────────────────────────┤
    │ 硬件层 (Hardware Layer):                                        │
    │  - 海康威视工业相机 (Hikvision Industrial Camera)               │
    │  - Jetson Orin NX 计算平台 (NVIDIA Jetson Computing Platform)   │
    │  - 串口通信模块 (Serial Communication Module)                   │
    ├─────────────────────────────────────────────────────────────────┤
    │ 图像处理层 (Image Processing Layer):                             │
    │  - 图像采集与预处理 (Image Acquisition & Preprocessing)         │
    │  - 目标矩形检测 (Target Rectangle Detection)                    │
    │  - 透视变换与校正 (Perspective Transform & Correction)          │
    │  - 距离估算 (Distance Estimation)                               │
    ├─────────────────────────────────────────────────────────────────┤
    │ 导航控制层 (Navigation Control Layer):                          │
    │  - 激光点映射 (Laser Point Mapping)                            │
    │  - 误差计算 (Error Calculation)                                │
    │  - 串口数据传输 (Serial Data Transmission)                      │
    │  - 实时调试接口 (Real-time Debug Interface)                     │
    └─────────────────────────────────────────────────────────────────┘

核心功能 (Core Features):
    1. 双矩形嵌套检测: 自动识别外框和内框矩形目标
    2. 透视变换校正: 将倾斜视角的目标校正为标准平面视图
    3. 基于面积的距离估算: 通过目标面积计算与目标的距离
    4. 激光点坐标映射: 根据距离信息映射激光点坐标
    5. 实时误差计算: 计算激光点与目标中心的偏差
    6. 串口数据传输: 将导航数据发送给下位机
    7. 可视化调试界面: 提供丰富的调试和标定工具

技术规格 (Technical Specifications):
    - 支持图像分辨率: 640x480 (可调整)
    - 检测精度: 亚像素级别
    - 距离测量范围: 500-1500mm
    - 通信协议: 自定义串口协议 (921600 bps)
    - 实时性能: > 30 FPS (Jetson Orin NX)

使用场景 (Use Cases):
    - 电子设计竞赛激光追踪任务
    - 机器人视觉导航
    - 目标跟踪与定位
    - 工业自动化视觉检测

更新日志 (Changelog):
v2.3 - 距离映射激光点版本
- 统一管理所有阈值调试变量到顶部区域 
- 为主图像预处理二值化创建全局调试变量
- 添加调试窗口显示控制开关
- 添加实时阈值调整功能 (+/- 调整主阈值, [/] 调整校正阈值)
- 增强键盘快捷键系统 (B:二值化 E:边缘 C:合并 D:全部调试 1:打印参数)
- 修复OpenCV窗口管理问题，避免销毁未创建的窗口错误
- 添加安全窗口管理函数，跟踪窗口创建状态
- 删除激光点检测功能，改为基于距离映射的激光点坐标系统
- 激光点坐标基于目标距离线性映射：最远处(1500mm+)坐标为(320,250)，最近处(500mm-)为(320,240)
- 只映射Y坐标，X坐标固定为320，实现线性关系映射

开源声明 (Open Source Declaration):
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
"""

# =============================================================================
# 导入库文件 (Import Libraries)
# =============================================================================
import sys              # 系统相关功能 (System utilities)
import time             # 时间处理 (Time handling)
import cv2              # OpenCV计算机视觉库 (OpenCV computer vision library)
import numpy as np      # NumPy数值计算库 (NumPy numerical computing)
from ctypes import *    # C类型接口，用于SDK调用 (C types interface for SDK calls)
import os               # 操作系统接口 (Operating system interface)
import serial           # 串口通信库 (Serial communication library)
import struct           # 二进制数据打包 (Binary data packing)

# =============================================================================
# 硬件平台检测模块 (Hardware Platform Detection Module)
# 功能: 自动识别运行平台，为不同硬件平台选择合适的SDK路径
# =============================================================================
IS_JETSON = os.path.exists('/etc/nv_tegra_release')  # 检测是否为Jetson平台

# =============================================================================
# 海康威视相机SDK导入模块 (Hikvision Camera SDK Import Module)
# 功能: 
# 1. 自动检测不同平台的SDK安装路径
# 2. 动态导入海康威视相机控制类
# 3. 提供SDK可用性标志
# 支持平台: Jetson系列 (aarch64), x86/x64 Linux系统
# =============================================================================
try:
    # 根据硬件平台选择对应的SDK路径
    if IS_JETSON:
        # Jetson平台 (ARM64架构) SDK路径候选列表
        sdk_paths = [
            "/opt/MVS/Samples/aarch64/Python/MvImport",         # 标准安装路径
            "/usr/local/MVS/Samples/aarch64/Python/MvImport",   # 自定义安装路径
            "/home/nvidia/MVS/Samples/aarch64/Python/MvImport"  # 用户目录安装
        ]
    else:
        # x86/x64平台SDK路径候选列表
        sdk_paths = [
            "/opt/MVS/Samples/Python/MvImport",         # 标准安装路径
            "/usr/local/MVS/Samples/Python/MvImport"    # 自定义安装路径
        ]
    
    # 搜索并添加有效的SDK路径到Python路径
    sdk_found = False
    for path in sdk_paths:
        if os.path.exists(path):
            sys.path.append(path)
            sdk_found = True
            break
    
    # 如果标准路径都不存在，尝试当前目录的MvImport文件夹
    if not sdk_found:
        sys.path.append("./MvImport")
    
    # 导入海康威视相机控制类
    # 包含: MvCamera, MV_CC_DEVICE_INFO, 各种常量和结构体定义
    from MvCameraControl_class import *
    SDK_OK = True  # SDK导入成功标志
except ImportError:
    SDK_OK = False  # SDK导入失败标志，系统将以兼容模式运行

# =============================================================================
# 系统全局变量定义 (System Global Variables Definition)
# 管理整个视觉系统的状态和数据缓存
# =============================================================================

# --- 相机控制模块全局变量 (Camera Control Module Variables) ---
cam = None              # 海康威视相机对象实例 (Hikvision camera instance)
device_list = None      # 设备列表缓存 (Device list cache)
connected = False       # 相机连接状态标志 (Camera connection status flag)
data_buf = None         # 图像数据缓冲区 (Image data buffer)
payload_size = 0        # 图像数据负载大小 (Image payload size)

# --- 串口通信模块全局变量 (Serial Communication Module Variables) ---
serial_port = None      # 串口对象实例 (Serial port instance)
serial_enabled = False  # 串口使能状态 (Serial port enable status)
selected_port = None    # 选定的串口设备名 (Selected serial port name)
selected_baudrate = None # 选定的波特率 (Selected baudrate)

# --- 性能监控模块全局变量 (Performance Monitoring Module Variables) ---
frame_count = 0         # 帧计数器，用于FPS计算 (Frame counter for FPS calculation)
start_time = time.time() # 系统启动时间戳 (System startup timestamp)

# --- 交互界面模块全局变量 (Interactive Interface Module Variables) ---
clicked_point = None    # 用户鼠标点击坐标 (User mouse click coordinates)
final_center_point = (-1, -1) # 最终检测到的目标中心点 (Final detected target center)

# --- 图像处理模块全局变量 (Image Processing Module Variables) ---
corrected_image = None  # 透视校正后的图像 (Perspective corrected image)
M = None               # 透视变换矩阵 (Perspective transform matrix)
M_inv = None           # 逆透视变换矩阵 (Inverse perspective transform matrix)
corrected_size = None   # 校正图像尺寸 (Corrected image size)
center_in_corrected = None # 校正图像中的目标中心点 (Target center in corrected image)

# =============================================================================
# 图像处理算法参数配置中心 (Image Processing Algorithm Parameter Center)
# 功能: 集中管理所有影响图像处理和检测精度的关键参数
# 优势: 便于调试、标定和性能优化，支持实时参数调整
# =============================================================================

# --- 图像预处理模块参数 (Image Preprocessing Module Parameters) ---
BINARY_THRESHOLD = 35              # 主图像二值化阈值 (0-255)
                                   # 说明: 控制黑白分割的灰度临界值，较小值检测更多黑色区域
                                   # 调整: 目标过暗时减小，目标过亮时增大
                                   
BINARY_THRESHOLD_CORRECTED = 35    # 校正图像二值化阈值 (0-255)
                                   # 说明: 透视校正后图像的二值化阈值，通常与主阈值相同
                                   # 调整: 校正后图像对比度变化时需要独立调整
                                   
CANNY_LOWER = 50                   # Canny边缘检测下阈值 (0-255)
                                   # 说明: 边缘检测的弱边缘阈值，影响边缘连续性
                                   # 调整: 噪声多时增大，边缘断裂时减小
                                   
CANNY_UPPER = 150                  # Canny边缘检测上阈值 (0-255)
                                   # 说明: 边缘检测的强边缘阈值，影响边缘敏感度
                                   # 调整: 一般为下阈值的2-3倍
                                   
GAUSSIAN_BLUR_SIZE = 5             # 高斯模糊核大小 (奇数, >=3)
                                   # 说明: 预处理降噪的模糊程度，影响边缘平滑度
                                   # 调整: 噪声多时增大，细节重要时减小

# --- 可视化调试开关 (Visualization Debug Switches) ---
DEBUG_SHOW_BINARY = True           # 显示二值化处理结果窗口
DEBUG_SHOW_EDGES = False           # 显示边缘检测结果窗口  
DEBUG_SHOW_COMBINED = False        # 显示合并预处理结果窗口

# --- 几何形状检测参数 (Geometric Shape Detection Parameters) ---
ANGLE_TOLERANCE = 30               # 矩形角度容差 (度, 0-45)
                                   # 说明: 判断四边形是否为矩形的角度误差允许范围
                                   # 调整: 目标倾斜严重时增大，精度要求高时减小
                                   
SIDE_RATIO_TOLERANCE = 0.4         # 矩形边长比容差 (0-1)
                                   # 说明: 对边长度比的误差允许范围
                                   # 调整: 目标变形大时增大，要求严格时减小
                                   
MIN_CONTOUR_AREA = 1000            # 最小轮廓面积 (像素²)
                                   # 说明: 过滤小噪声的面积下限
                                   # 调整: 根据目标距离和图像分辨率调整
                                   
MAX_CONTOUR_AREA = 307200          # 最大轮廓面积 (像素²)
                                   # 说明: 过滤过大区域的面积上限 (640×480=307200)
                                   # 调整: 通常设为图像总面积
                                   
MIN_PERIMETER = 20                 # 最小轮廓周长 (像素)
                                   # 说明: 过滤点状噪声的周长下限
                                   
APPROX_EPSILON_FACTOR = 0.02       # 轮廓近似精度因子 (0-0.1)
                                   # 说明: Douglas-Peucker算法的近似精度系数
                                   # 调整: 要求精确时减小，允许近似时增大
                                   
MIN_AREA_RATIO = 0.7               # 嵌套矩形最小面积比 (0-1)
                                   # 说明: 内外矩形面积比的下限，用于过滤干扰项
                                   # 调整: 内矩形相对较大时减小，严格过滤时增大

# --- 距离测量模块参数 (Distance Measurement Module Parameters) ---
REFERENCE_AREA_PIXELS = 8412       # 参考物体在基准距离下的像素面积
                                   # 说明: 用于面积-距离换算的标定基准
                                   # 标定方法: 将A4纸放在1500mm处，记录检测面积
                                   
REFERENCE_DISTANCE_MM = 1500       # 参考标定距离 (毫米)
                                   # 说明: 进行面积标定时的物理距离
                                   
REAL_OBJECT_AREA_MM2 = 62370      # 实际物体面积 (平方毫米)
                                   # 说明: A4纸实际面积 210mm × 297mm = 62370 mm²

# =============================================================================
# 激光点映射导航系统参数 (Laser Point Mapping Navigation System Parameters)
# 功能: 基于目标距离信息，将激光点映射到预设的坐标系统中
# 原理: 采用线性映射算法，将物理距离转换为屏幕坐标，实现激光导航
# =============================================================================

# --- 距离映射范围定义 (Distance Mapping Range Definition) ---
MAX_DISTANCE_MM = 1500             # 最远有效测距距离 (毫米)
                                   # 说明: 超过此距离时激光点坐标固定在最远位置
                                   # 应用: 防止超距离时的坐标异常跳变
                                   
MIN_DISTANCE_MM = 500              # 最近有效测距距离 (毫米)  
                                   # 说明: 小于此距离时激光点坐标固定在最近位置
                                   # 应用: 避免过近距离导致的测量误差

# --- 激光点坐标映射参数 (Laser Point Coordinate Mapping Parameters) ---
LASER_X_FIXED = 322                # 激光点X坐标固定值 (像素)
                                   # 说明: X坐标不随距离变化，对应图像中心偏右
                                   # 设计原因: 激光器安装位置决定的固定偏移
                                   
LASER_Y_MAX_DISTANCE = 250         # 最远距离对应的Y坐标 (像素)
                                   # 说明: 当目标距离≥1500mm时，激光点Y坐标为250
                                   # 映射逻辑: 距离越远，激光点越靠下
                                   
LASER_Y_MIN_DISTANCE = 240         # 最近距离对应的Y坐标 (像素)
                                   # 说明: 当目标距离≤500mm时，激光点Y坐标为240
                                   # 映射逻辑: 距离越近，激光点越靠上
                                   
# 线性映射公式: Y = Y_MIN + (distance - MIN_DISTANCE) * (Y_MAX - Y_MIN) / (MAX_DISTANCE - MIN_DISTANCE)
# 示例: 距离1000mm时，Y = 240 + (1000-500) * (250-240) / (1500-500) = 245

# =============================================================================
# 调试可视化窗口管理系统 (Debug Visualization Window Management System)  
# 功能: 安全管理OpenCV调试窗口的创建和销毁，防止重复创建或销毁未存在窗口
# 解决问题: OpenCV在销毁未创建窗口时会产生错误，影响系统稳定性
# =============================================================================
windows_created = {
    'binary': False,      # 二值化处理结果显示窗口状态
    'edges': False,       # 边缘检测结果显示窗口状态  
    'combined': False     # 合并预处理结果显示窗口状态
}

# =============================================================================
# HSV颜色空间分析调试系统 (HSV Color Space Analysis Debug System)
# 功能: 提供交互式HSV颜色信息分析工具，辅助激光检测参数调整
# 使用方法: 在主窗口点击任意位置，系统自动分析该点及周围区域的HSV信息
# 应用场景: 激光点检测阈值设定、环境光影响分析、颜色分割参数优化
# =============================================================================
current_frame_for_hsv = None    # 当前帧的HSV分析用图像缓存
current_mouse_pos = None        # 实时鼠标位置坐标
hsv_output_requested = False    # HSV信息输出请求标志

# =============================================================================
# 调试参数实时调整说明 (Real-time Debug Parameter Adjustment Guide)
# 
# 键盘快捷键功能映射:
# ├── 显示控制键 (Display Control Keys):
# │   ├── 'B' : 切换二值化结果显示 (Toggle Binary Result Display)
# │   ├── 'E' : 切换边缘检测结果显示 (Toggle Edge Detection Display) 
# │   ├── 'C' : 切换合并结果显示 (Toggle Combined Result Display)
# │   └── 'D' : 切换所有调试显示 (Toggle All Debug Display)
# │
# ├── 阈值调整键 (Threshold Adjustment Keys):
# │   ├── '+'/'-' : 调整主二值化阈值 ±5 (Main Binary Threshold ±5)
# │   ├── '['/']' : 调整校正图像二值化阈值 ±5 (Corrected Binary Threshold ±5)
# │   └── 'A'/'Z' : 调整面积比过滤阈值 ±0.05 (Area Ratio Threshold ±0.05)
# │
# ├── 功能控制键 (Function Control Keys):
# │   ├── 'S' : 保存当前图像 (Save Current Images)
# │   ├── 'R' : 重置点击点 (Reset Click Point)
# │   ├── 'L' : 显示激光映射参数 (Show Laser Mapping Parameters)
# │   ├── '1' : 打印所有当前参数 (Print All Current Parameters)
# │   └── 'Q' : 退出系统 (Quit System)
# │
# └── 交互分析 (Interactive Analysis):
#     └── 鼠标点击 : 获取点击位置HSV信息 (Get HSV Info at Click Position)
#
# 参数调整策略 (Parameter Adjustment Strategy):
# 1. 二值化阈值调整原则:
#    - 目标过暗不易检测 → 降低阈值
#    - 背景噪声过多 → 提高阈值
#    - 激光干扰严重 → 配合闭运算参数调整
#
# 2. 面积比阈值调整原则:  
#    - 内外矩形大小相近 → 降低阈值(0.5-0.7)
#    - 内矩形明显较小 → 提高阈值(0.7-0.9)
#    - 干扰项较多 → 提高阈值增强过滤
#
# 3. 边缘检测阈值调整原则:
#    - 边缘断裂 → 降低上下阈值
#    - 噪声边缘过多 → 提高上下阈值  
#    - 保持上下阈值比例在2:1-3:1之间
# =============================================================================

def estimate_distance_by_area(area_pixels):
    """
    基于面积法估算目标物理距离 (Distance Estimation Using Area Method)
    
    功能原理 (Function Principle):
        采用反比例函数模型，基于物体在不同距离下的成像面积变化规律进行距离估算。
        物理基础：当相机参数固定时，物体在图像中的投影面积与距离的平方成反比关系。
        
    数学模型 (Mathematical Model):
        S₁/S₂ = (D₂/D₁)²  =>  D₂ = D₁ × √(S₁/S₂)
        其中：S₁为参考面积，D₁为参考距离，S₂为当前面积，D₂为待求距离
        
    标定方法 (Calibration Method):
        1. 将标准目标(A4纸)放置在已知距离(1500mm)处
        2. 记录此时检测到的像素面积作为参考面积
        3. 系统将基于此参考值计算其他距离下的实际距离
        
    精度影响因素 (Accuracy Factors):
        - 目标检测精度：轮廓提取的准确性直接影响面积计算
        - 透视畸变：相机镜头畸变会影响面积测量精度
        - 光照条件：光照变化会影响目标边界识别
        - 目标姿态：目标倾斜会改变投影面积
        
    Args:
        area_pixels (int): 检测到的目标物体像素面积
                          取值范围：> 0，通常在 1000-50000 像素²
                          
    Returns:
        float: 估算的物理距离 (毫米)
               -1: 输入参数无效 (面积 ≤ 0)
               正值: 有效的距离估算结果
               
    使用示例 (Usage Example):
        >>> area = 8412  # 检测到的目标面积
        >>> distance = estimate_distance_by_area(area)
        >>> print(f"估算距离: {distance:.1f}mm")
        估算距离: 1500.0mm
        
    注意事项 (Important Notes):
        - 确保参考参数(REFERENCE_AREA_PIXELS, REFERENCE_DISTANCE_MM)已正确标定
        - 该方法假设目标为平面且垂直于相机光轴
        - 距离估算精度随目标距离增加而降低
    """
    # 输入参数有效性检查 - 面积必须为正数
    if area_pixels <= 0:
        return -1  # 返回-1表示无效输入
    
    # 应用面积-距离反比例公式进行距离估算
    # 公式: D = D_ref × √(A_ref / A_current)
    # 其中: D_ref=1500mm, A_ref=8412像素²
    distance_mm = REFERENCE_DISTANCE_MM * np.sqrt(REFERENCE_AREA_PIXELS / area_pixels)
    
    return distance_mm  # 返回估算距离(毫米)

def print_calibration_info(area_pixels, distance_mm):
    """
    打印标定信息，帮助用户调整参考参数
    
    Args:
        area_pixels: 当前检测到的像素面积
        distance_mm: 估算的距离
    """
    print(f"========== 距离估计标定信息 ==========")
    print(f"当前检测面积: {area_pixels:.0f} 像素")
    print(f"估算距离: {distance_mm:.1f} mm ({distance_mm/10:.1f} cm)")
    print(f"参考面积: {REFERENCE_AREA_PIXELS} 像素 @ {REFERENCE_DISTANCE_MM} mm")
    print(f"实际物体面积: {REAL_OBJECT_AREA_MM2} mm² (A4纸)")
    print("标定说明:")
    print("1. 将A4纸放在已知距离处")
    print("2. 记录检测到的像素面积")
    print("3. 修改代码中的REFERENCE_AREA_PIXELS和REFERENCE_DISTANCE_MM")
    print("=====================================")

def safe_decode_string(byte_array):
    """安全解码字节数组为字符串"""
    try:
        if hasattr(byte_array, 'decode'):
            return byte_array.decode('utf-8', errors='ignore')
        elif hasattr(byte_array, 'value'):
            return byte_array.value.decode('utf-8', errors='ignore')
        elif isinstance(byte_array, (bytes, bytearray)):
            return byte_array.decode('utf-8', errors='ignore')
        else:
            if hasattr(byte_array, '__len__'):
                byte_data = bytes([b for b in byte_array if b != 0])
                return byte_data.decode('utf-8', errors='ignore')
            else:
                return str(byte_array)
    except Exception:
        return "Unknown Device"

def calculate_laser_position_by_distance(distance_mm):
    """
    基于目标距离映射激光点导航坐标 (Calculate Laser Navigation Coordinates by Distance)
    
    功能描述 (Function Description):
        将物理距离测量值转换为激光点在导航坐标系中的位置，实现基于距离的激光导航。
        该函数是激光追踪小车导航系统的核心算法之一，提供稳定可靠的激光点位置映射。
        
    映射原理 (Mapping Principle):
        采用线性映射算法，将连续的物理距离映射到离散的像素坐标系统：
        - X轴坐标固定：由激光器的物理安装位置决定，不随距离变化
        - Y轴坐标线性变化：距离越远，Y坐标越大（激光点越靠近图像下方）
        
    线性映射公式 (Linear Mapping Formula):
        Y = Y_min + (D_clamped - D_min) × (Y_max - Y_min) / (D_max - D_min)
        其中：
        - D_clamped: 限制在有效范围内的距离值
        - Y_min/Y_max: 最近/最远距离对应的Y坐标
        - D_min/D_max: 最近/最远有效距离阈值
        
    坐标系统说明 (Coordinate System):
        - 原点位置：图像左上角 (0, 0)
        - X轴方向：从左到右递增
        - Y轴方向：从上到下递增
        - 激光点X坐标固定在图像中心偏右位置
        
    Args:
        distance_mm (float): 测量得到的目标距离 (毫米)
                           取值范围：任意正数值
                           有效范围：500-1500mm
                           超出范围会被自动限制
                           
    Returns:
        tuple: 激光点在图像坐标系中的位置 (x, y)
               x (int): X坐标，固定值 322
               y (int): Y坐标，根据距离线性映射，范围 240-250
               
    映射示例 (Mapping Examples):
        >>> calculate_laser_position_by_distance(500)   # 最近距离
        (322, 240)
        >>> calculate_laser_position_by_distance(1000)  # 中等距离  
        (322, 245)
        >>> calculate_laser_position_by_distance(1500)  # 最远距离
        (322, 250)
        >>> calculate_laser_position_by_distance(2000)  # 超出范围，限制到最远
        (322, 250)
        
    设计考虑 (Design Considerations):
        1. 距离限制：防止异常距离值导致坐标越界
        2. 线性映射：确保平滑的激光点移动轨迹
        3. 整数坐标：避免亚像素坐标带来的显示问题
        4. 固定X坐标：简化导航控制逻辑
        
    应用场景 (Use Cases):
        - 激光追踪小车的实时导航
        - 可视化调试中的激光点显示
        - 导航误差计算的基准坐标
    """
    # 步骤1: 设置X坐标为固定值(基于激光器安装位置)
    x = LASER_X_FIXED  # X = 322像素
    
    # 步骤2: 限制距离到有效范围内，防止越界
    distance_clamped = max(MIN_DISTANCE_MM, min(MAX_DISTANCE_MM, distance_mm))
    # 限制范围: 500mm ≤ distance ≤ 1500mm
    
    # 步骤3: 线性映射计算Y坐标 
    # 公式: Y = Y_min + (D - D_min) × (Y_max - Y_min) / (D_max - D_min)
    # 500mm→240像素, 1500mm→250像素, 线性插值
    y = LASER_Y_MIN_DISTANCE + (distance_clamped - MIN_DISTANCE_MM) * \
        (LASER_Y_MAX_DISTANCE - LASER_Y_MIN_DISTANCE) / (MAX_DISTANCE_MM - MIN_DISTANCE_MM)
    
    # 步骤4: 转换为整数坐标并返回
    return (int(x), int(y))  # 返回(322, 240-250)范围内的坐标

def draw_laser_position_info(image, laser_pt, distance_mm):
    """在图像上绘制激光点位置信息"""
    if laser_pt is not None:
        # 绘制激光点
        cv2.circle(image, laser_pt, 10, (0, 255, 255), 2)  # 黄色圆圈
        cv2.circle(image, laser_pt, 4, (0, 0, 255), -1)   # 红色实心圆
        
        # 绘制十字线
        cross_size = 20
        cv2.line(image, (laser_pt[0] - cross_size, laser_pt[1]), 
                 (laser_pt[0] + cross_size, laser_pt[1]), (0, 255, 255), 2)
        cv2.line(image, (laser_pt[0], laser_pt[1] - cross_size), 
                 (laser_pt[0], laser_pt[1] + cross_size), (0, 255, 255), 2)
        
        # 标注坐标和距离
        label = f"Laser: ({laser_pt[0]}, {laser_pt[1]}) | {distance_mm:.1f}mm"
        label_pos = (laser_pt[0] + 25, laser_pt[1] - 25)
        cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 在图像角落显示映射状态
        cv2.putText(image, "LASER MAPPED", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return True
    else:
        # 显示未映射激光点
        cv2.putText(image, "NO LASER MAPPING", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False

def calculate_laser_error(laser_pt, target_center, pixels_per_mm=1.5):
    """计算激光点相对于目标中心的误差"""
    if laser_pt is None or target_center is None:
        return None, None, None
    
    # 计算像素差值
    dx_pixels = laser_pt[0] - target_center[0]
    dy_pixels = laser_pt[1] - target_center[1]
    
    # 转换为毫米
    dx_mm = dx_pixels / pixels_per_mm
    dy_mm = dy_pixels / pixels_per_mm
    
    # 计算距离
    distance_pixels = np.sqrt(dx_pixels**2 + dy_pixels**2)
    distance_mm = distance_pixels / pixels_per_mm
    
    return dx_mm, dy_mm, distance_mm

def safe_show_window(window_name, image):
    """安全显示窗口，跟踪窗口状态"""
    global windows_created
    cv2.imshow(window_name, image)
    windows_created[window_name] = True

def safe_destroy_window(window_name):
    """安全销毁窗口，只销毁已创建的窗口"""
    global windows_created
    if windows_created.get(window_name, False):
        cv2.destroyWindow(window_name)
        windows_created[window_name] = False

def cleanup_all_debug_windows():
    """清理所有调试窗口"""
    global windows_created
    for window_name in windows_created.keys():
        safe_destroy_window(window_name)

def select_serial_port():
    """选择串口 - 固定使用 /dev/ttyTHS1 @ 921600"""
    global selected_port, selected_baudrate
    
    selected_port = "/dev/ttyTHS1"
    selected_baudrate = 921600
    
    print(f"✓ 串口配置: {selected_port} @ {selected_baudrate}")
    return True

def test_serial_communication(port, baudrate):
    """测试串口通信"""
    print(f"\n正在测试串口通信: {port} @ {baudrate}")
    
    try:
        # 打开串口
        ser = serial.Serial(port, baudrate, timeout=1.0)
        print("✓ 串口打开成功")
        
        # 发送测试数据
        test_data = b'\xA5\x0A\x01\x00\x00\x00\x00\x00\x00\x00'  # 简单的测试帧
        ser.write(test_data)
        print("✓ 测试数据发送成功")
        
        # 尝试读取响应（可能没有响应，这是正常的）
        time.sleep(0.1)
        if ser.in_waiting > 0:
            response = ser.read(ser.in_waiting)
            print(f"✓ 收到响应: {response.hex()}")
        else:
            print("- 未收到响应（正常情况）")
        
        ser.close()
        print("✓ 串口通信测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 串口通信测试失败: {e}")
        return False

def setup_serial():
    """设置串口"""
    # 选择串口
    if not select_serial_port():
        print("❌ 未选择串口，将在无串口模式下运行")
        return False
    
    # 测试串口通信
    if not test_serial_communication(selected_port, selected_baudrate):
        retry = input("串口测试失败，是否继续? (y/n): ").strip().lower()
        if retry != 'y':
            return False
    
    # 初始化串口
    return init_serial()

def init_serial():
    """初始化串口"""
    global serial_port, serial_enabled
    
    try:
        serial_port = serial.Serial(selected_port, selected_baudrate, timeout=0.1)
        serial_enabled = True
        print(f"✓ 串口 {selected_port} @ {selected_baudrate} 初始化成功")
        return True
    except Exception as e:
        print(f"❌ 串口初始化失败: {e}")
        serial_enabled = False
        return False

def pack_frame(cmd_id, flags, floats):
    """
    自定义串口通信协议数据帧打包器 (Custom Serial Communication Protocol Frame Packer)
    
    协议设计 (Protocol Design):
        专为电赛激光追踪小车设计的高效二进制通信协议，支持实时导航数据传输。
        
    帧结构定义 (Frame Structure Definition):
        ┌─────────┬─────────┬─────────┬─────────┬─────────────────┐
        │ 帧头    │ 长度    │ 命令ID  │ 标志位  │ 浮点数据        │
        │ Header  │ Length  │ CmdID   │ Flags   │ Float Data      │
        ├─────────┼─────────┼─────────┼─────────┼─────────────────┤
        │ 1 Byte  │ 1 Byte  │ 2 Bytes │ 2 Bytes │ N×4 Bytes       │
        │ 0xA5    │ 6+4×N   │ uint16  │ uint16  │ float32 array   │
        └─────────┴─────────┴─────────┴─────────┴─────────────────┘
        
    字节序说明 (Endianness):
        采用小端序(Little Endian)编码，兼容ARM和x86架构
        
    协议特点 (Protocol Features):
        - 固定帧头：0xA5，便于帧同步和错误检测
        - 动态长度：支持可变数量的浮点数据
        - 类型标识：命令ID和标志位提供数据类型信息
        - 高效编码：二进制格式，传输效率高
        
    数据类型映射 (Data Type Mapping):
        - 帧头(Header): uint8, 固定0xA5
        - 长度(Length): uint8, 数据部分字节数 
        - 命令ID(CmdID): uint16, 区分不同类型的数据
        - 标志位(Flags): uint16, 扩展信息和状态标记
        - 浮点数据: float32 array, IEEE 754标准
        
    应用场景 (Application Scenarios):
        - 激光点坐标传输
        - 导航误差数据传输  
        - 系统状态信息传输
        - 实时控制指令传输
        
    Args:
        cmd_id (int): 命令标识符 (0-65535)
                     用于区分不同类型的数据包
                     例如: 0x0100-点击数据, 0x0101-激光数据
                     
        flags (int): 标志位 (0-65535) 
                    用于传递额外的状态和控制信息
                    可用于数据优先级、错误标记等
                    
        floats (list): 浮点数据列表
                      包含需要传输的实际数值数据
                      通常为坐标、误差、距离等测量值
                      
    Returns:
        bytes: 打包后的二进制数据帧
              可直接通过串口发送的字节序列
              
    数据完整性 (Data Integrity):
        - 固定帧头提供基本同步检测
        - 长度字段支持帧边界识别  
        - 可扩展校验和或CRC机制
        
    性能指标 (Performance Metrics):
        - 打包速度: < 1ms (典型12浮点数)
        - 传输效率: 90%+ (有效数据/总数据)
        - 兼容性: 支持多种处理器架构
        
    使用示例 (Usage Example):
        >>> data = [1.5, 2.3, 0.8, 1234567.0]  
        >>> frame = pack_frame(0x0101, 0x0001, data)
        >>> print(f"数据帧长度: {len(frame)} 字节")
        数据帧长度: 22 字节
    """
    # 计算浮点数数量和总长度
    n = len(floats)
    length = 2 + 2 + 4 * n  # 命令ID(2) + 标志位(2) + 浮点数据(4×n)
    
    # 创建字节缓冲区
    buf = bytearray()
    
    # 添加帧头标识 (0xA5)
    buf.append(0xA5)
    
    # 添加数据长度 (不包含帧头和长度字段本身)
    buf.append(length & 0xFF)
    
    # 添加命令ID (小端序16位)
    buf += struct.pack('<H', cmd_id)
    
    # 添加标志位 (小端序16位)  
    buf += struct.pack('<H', flags)
    
    # 添加浮点数据数组 (小端序32位浮点)
    for f in floats:
        buf += struct.pack('<f', f)
    
    # 返回打包后的字节序列
    return bytes(buf)

def send_error_data(dx_mm, dy_mm, distance_mm, data_type="click"):
    """发送误差数据到串口"""
    global serial_port, serial_enabled
    
    if not serial_enabled or serial_port is None:
        print("⚠️ 串口未启用，无法发送数据")
        return False
    
    # if distance_mm is None:
    #     distance_mm = np.sqrt(dx_mm**2 + dy_mm**2)
    
    # 根据数据类型设置不同的命令ID
    if data_type == "laser":
        cmd_id = 0x0101  # 当前激光点数据
        flags = 0x0001
    elif data_type == "last_laser":
        cmd_id = 0x0102  # 最后激光点数据
        flags = 0x0002
    else:
        cmd_id = 0x0100  # 其他数据
        flags = 0x0000
    
    error_data = [
        float(dx_mm),
        float(dy_mm), 
        float(distance_mm),
        float(time.time())
    ] + [0.0] * 8  # 后8位用float占位
    
    try:
        frame_data = pack_frame(cmd_id=cmd_id, flags=flags, floats=error_data)
        serial_port.write(frame_data)
        # print(f"✓ 发送{data_type}误差数据: X={dx_mm:.1f}mm, Y={dy_mm:.1f}mm, D={distance_mm:.1f}mm")
        return True
    except Exception as e:
        print(f"❌ 串口发送失败: {e}")
        return False

def list_devices():
    """列出所有设备"""
    global device_list
    
    if not SDK_OK:
        return []
        
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    
    if ret != 0:
        print(f"枚举设备失败，错误码: {ret}")
        return []
    
    if device_list.nDeviceNum == 0:
        print("未找到任何设备")
        return []
    
    devices = []
    print(f"找到 {device_list.nDeviceNum} 个设备")
    
    for i in range(device_list.nDeviceNum):
        device_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        
        if device_info.nTLayerType == MV_GIGE_DEVICE:
            gige_info = cast(byref(device_info.SpecialInfo.stGigEInfo), POINTER(MV_GIGE_DEVICE_INFO)).contents
            
            device_name = safe_decode_string(gige_info.chUserDefinedName)
            if not device_name.strip():
                device_name = safe_decode_string(gige_info.chModelName)
            if not device_name.strip():
                device_name = f"GigE_Device_{i}"
            
            ip_addr = f"{(gige_info.nCurrentIp >> 0) & 0xFF}.{(gige_info.nCurrentIp >> 8) & 0xFF}.{(gige_info.nCurrentIp >> 16) & 0xFF}.{(gige_info.nCurrentIp >> 24) & 0xFF}"
            device_name = f"GigE-{i}: {device_name} ({ip_addr})"
                    
        elif device_info.nTLayerType == MV_USB_DEVICE:
            usb_info = cast(byref(device_info.SpecialInfo.stUsb3VInfo), POINTER(MV_USB3_DEVICE_INFO)).contents
            
            device_name = safe_decode_string(usb_info.chUserDefinedName)
            if not device_name.strip():
                device_name = safe_decode_string(usb_info.chModelName)
            if not device_name.strip():
                device_name = f"USB_Device_{i}"
            
            device_name = f"USB-{i}: {device_name}"
            
        else:
            device_name = f"Unknown-{i}: Device_{i}"
        
        devices.append(device_name)
        print(f"  设备 {i}: {device_name}")
    
    return devices

def connect_camera(device_index=0):
    """连接指定设备"""
    global cam, connected, payload_size, data_buf
    
    if not device_list or device_index >= device_list.nDeviceNum:
        print("设备索引无效")
        return False
        
    print(f"正在连接设备 {device_index}...")
    
    cam = MvCamera()
    device_info = cast(device_list.pDeviceInfo[device_index], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(device_info)
    if ret != 0:
        print(f"创建句柄失败，错误码: {ret}")
        return False
        
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"打开设备失败，错误码: {ret}")
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Monitor, 0)
        if ret != 0:
            print(f"共享模式打开也失败，错误码: {ret}")
            return False
        else:
            print("⚠️ 以监控模式打开设备")
    
    pixel_format = MVCC_ENUMVALUE()
    memset(byref(pixel_format), 0, sizeof(MVCC_ENUMVALUE))
    ret = cam.MV_CC_GetEnumValue("PixelFormat", pixel_format)
    if ret == 0:
        print(f"当前像素格式: 0x{pixel_format.nCurValue:08X}")
    
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    
    if device_info.nTLayerType == MV_GIGE_DEVICE:
        cam.MV_CC_SetIntValue("GevSCPSPacketSize", 1500)
        print("✓ 设置GigE包大小")
    
    param = MVCC_INTVALUE()
    memset(byref(param), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("PayloadSize", param)
    if ret == 0:
        payload_size = param.nCurValue
        print(f"PayloadSize: {payload_size}")
    else:
        payload_size = 1920 * 1080 * 3
        print(f"使用默认PayloadSize: {payload_size}")
    
    connected = True
    print("✓ 设备连接成功")
    return True

def start_capture():
    """开始采集"""
    global frame_count, start_time
    
    if not connected:
        print("设备未连接")
        return False
    
    ret = cam.MV_CC_StartGrabbing()
    if ret == 0:
        print("✓ 开始采集")
        frame_count = 0
        start_time = time.time()
        return True
    else:
        print(f"开始采集失败，错误码: {ret}")
        return False

def get_image():
    """获取一帧图像"""
    global data_buf, frame_count
    
    if not connected:
        return None
        
    if data_buf is None:
        data_buf = (c_ubyte * payload_size)()
    
    frame_info = MV_FRAME_OUT_INFO_EX()
    memset(byref(frame_info), 0, sizeof(frame_info))
    
    ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), payload_size, frame_info, 2000)
    if ret != 0:
        if ret == MV_E_NODATA:
            return None
        else:
            print(f"获取图像失败，错误码: {ret}")
            return None
    
    image_data = np.frombuffer(data_buf, count=int(frame_info.nFrameLen), dtype=np.uint8)
    height, width = frame_info.nHeight, frame_info.nWidth
    
    if len(image_data) == 0:
        return None
    
    bayer_image = image_data.reshape((height, width))
    image = cv2.cvtColor(bayer_image, cv2.COLOR_BayerRG2RGB)
    frame_count += 1
    return image

def get_fps():
    """获取当前FPS"""
    global frame_count, start_time
    elapsed = time.time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0

def set_exposure(exposure_time):
    """设置曝光时间(微秒)"""
    global current_exposure_time
    if not connected:
        return False
    # 获取当前曝光时间
    if 'current_exposure_time' not in globals():
        current_exposure_time = None
    # 如果传入的曝光时间与当前值相同，直接返回
    if current_exposure_time == exposure_time:
        return True
    # 设置新的曝光时间
    ret = cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time))
    if ret == 0:
        current_exposure_time = exposure_time
        print(f"✓ 曝光时间已设置为: {exposure_time}μs")
        # 延迟等待设置生效
        time.sleep(0.020)
        return True
    else:
        print(f"❌ 设置曝光时间失败，错误码: {ret}")
        return False

def set_gain(gain):
    """设置增益"""
    if connected:
        ret = cam.MV_CC_SetFloatValue("Gain", float(gain))
        return ret == 0
    return False

def set_framerate(fps):
    """设置帧率"""
    if connected:
        cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
        ret = cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
        return ret == 0
    return False

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数 - 增加HSV信息输出"""
    global clicked_point, current_mouse_pos, hsv_output_requested
    
    # 更新鼠标位置
    current_mouse_pos = (x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        hsv_output_requested = True  # 请求在主循环中输出HSV信息
        print(f"✓ 鼠标点击坐标: ({x}, {y})")

def output_hsv_info(image, x, y, region_size=5):
    """
    交互式HSV颜色空间分析工具 (Interactive HSV Color Space Analysis Tool)
    
    功能描述 (Function Description):
        对指定坐标点及其周围区域进行详细的HSV颜色空间分析，为激光检测、
        目标分割等计算机视觉任务提供精确的颜色参数参考。
        
    HSV颜色空间说明 (HSV Color Space Description):
        HSV(Hue, Saturation, Value)颜色空间更符合人类视觉感知：
        - H(色调): 0-179, 表示颜色种类(红、绿、蓝等)
        - S(饱和度): 0-255, 表示颜色纯度(灰色 → 纯色)  
        - V(明度): 0-255, 表示颜色亮度(黑色 → 白色)
        
    分析内容 (Analysis Content):
        1. 点击位置精确HSV值
        2. 周围区域HSV统计信息(最值、均值)
        3. 建议的HSV检测阈值范围  
        4. 红色激光特殊处理建议
        
    阈值建议策略 (Threshold Suggestion Strategy):
        - 基于区域统计扩展容差范围
        - 考虑光照变化的鲁棒性
        - 针对红色跨0度问题提供双区间方案
        
    调试应用场景 (Debug Application Scenarios):
        - 激光点检测参数调优
        - 环境光照影响评估
        - 颜色分割算法调试
        - 目标识别颜色特征提取
        
    参数调整指南 (Parameter Tuning Guide):
        1. H值稳定性检查: 重复点击同一激光点，观察H值变化
        2. S值阈值设定: 过低会包含背景，过高会遗漏目标
        3. V值环境适应: 需要考虑不同光照条件下的变化
        4. 容差范围平衡: 过小易丢失目标，过大易误检
    """
    try:
        height, width = image.shape[:2]
        
        # 检查坐标是否在图像范围内
        if x < 0 or x >= width or y < 0 or y >= height:
            print(f"⚠️ 坐标 ({x}, {y}) 超出图像范围")
            return
        
        # 转换为HSV颜色空间
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 获取点击点的HSV值
        point_hsv = hsv_image[y, x]
        h, s, v = point_hsv
        
        print(f"\n=== 点击位置 ({x}, {y}) HSV信息 ===")
        print(f"点击点HSV值: H={h}, S={s}, V={v}")
        
        # 获取周围区域的HSV范围
        half_region = region_size // 2
        y_start = max(0, y - half_region)
        y_end = min(height, y + half_region + 1)
        x_start = max(0, x - half_region)
        x_end = min(width, x + half_region + 1)
        
        region_hsv = hsv_image[y_start:y_end, x_start:x_end]
        
        if region_hsv.size > 0:
            # 计算区域HSV范围
            h_min, s_min, v_min = np.min(region_hsv, axis=(0, 1))
            h_max, s_max, v_max = np.max(region_hsv, axis=(0, 1))
            h_mean, s_mean, v_mean = np.mean(region_hsv, axis=(0, 1)).astype(int)
            
            print(f"周围{region_size}x{region_size}区域HSV范围:")
            print(f"  H范围: {h_min} - {h_max} (平均: {h_mean})")
            print(f"  S范围: {s_min} - {s_max} (平均: {s_mean})")
            print(f"  V范围: {v_min} - {v_max} (平均: {v_mean})")
            
            # 生成建议的HSV检测范围
            suggest_hsv_range(h_mean, s_mean, v_mean, h_min, h_max, s_min, s_max, v_min, v_max)
        
        print("=" * 40)
        
    except Exception as e:
        print(f"❌ 获取HSV信息失败: {e}")

def suggest_hsv_range(h_mean, s_mean, v_mean, h_min, h_max, s_min, s_max, v_min, v_max):
    """根据点击区域HSV值建议检测范围"""
    print(f"\n建议的HSV检测范围:")
    
    # HSV检测范围建议（适度放宽）
    h_tolerance = max(10, (h_max - h_min) + 5)
    s_tolerance = max(50, (s_max - s_min) + 20)
    v_tolerance = max(50, (v_max - v_min) + 20)
    
    suggested_h_min = max(0, h_mean - h_tolerance)
    suggested_h_max = min(179, h_mean + h_tolerance)
    suggested_s_min = max(0, s_mean - s_tolerance)
    suggested_s_max = min(255, s_mean + s_tolerance)
    suggested_v_min = max(0, v_mean - v_tolerance)
    suggested_v_max = min(255, v_mean + v_tolerance)
    
    print(f"  lower = np.array([{suggested_h_min}, {suggested_s_min}, {suggested_v_min}])")
    print(f"  upper = np.array([{suggested_h_max}, {suggested_s_max}, {suggested_v_max}])")
    
    # 特殊处理红色（跨越0度）
    if h_mean < 15 or h_mean > 165:
        print(f"\n检测到可能是红色激光，建议使用双范围:")
        if h_mean < 15:
            print(f"  范围1: lower1=np.array([0, {suggested_s_min}, {suggested_v_min}]), upper1=np.array([{h_mean + 10}, {suggested_s_max}, {suggested_v_max}])")
            print(f"  范围2: lower2=np.array([{max(170, h_mean + 170)}, {suggested_s_min}, {suggested_v_min}]), upper2=np.array([179, {suggested_s_max}, {suggested_v_max}])")
        else:
            print(f"  范围1: lower1=np.array([0, {suggested_s_min}, {suggested_v_min}]), upper1=np.array([{h_mean - 170 + 10}, {suggested_s_max}, {suggested_v_max}])")
            print(f"  范围2: lower2=np.array([{h_mean - 10}, {suggested_s_min}, {suggested_v_min}]), upper2=np.array([179, {suggested_s_max}, {suggested_v_max}])")

def preprocess_image(image):
    """
    多模态图像预处理算法 (Multi-Modal Image Preprocessing Algorithm)
    
    算法描述 (Algorithm Description):
        结合二值化处理和边缘检测的双通道预处理算法，专门针对激光追踪场景
        中的复杂光照条件和激光干扰问题进行优化设计。
        
    处理流程 (Processing Pipeline):
        输入彩色图像 → 灰度转换 → 二值化处理 → 形态学闭运算 → 边缘检测 → 通道融合 → 输出
        
    核心算法原理 (Core Algorithm Principles):
        
        1. 灰度转换 (Grayscale Conversion):
           采用加权平均法：Gray = 0.299R + 0.587G + 0.114B
           符合人眼视觉感知特性，保持亮度信息的准确性。
           
        2. 反向二值化 (Inverse Binary Thresholding):
           T(x,y) = { 255, if I(x,y) < threshold
                    { 0,   if I(x,y) ≥ threshold
           将黑色目标区域转换为白色前景，便于后续轮廓检测。
           
        3. 形态学闭运算 (Morphological Closing):
           Closing = Erosion(Dilation(Image))
           解决激光照亮黑色边框造成的断裂问题：
           - 先膨胀：连接断裂的边框线条
           - 后腐蚀：恢复目标的原始尺寸
           
        4. Canny边缘检测 (Canny Edge Detection):
           双阈值边缘检测算法：
           - 高阈值：检测强边缘，确保边缘真实性
           - 低阈值：连接弱边缘，保持边缘连续性
           
        5. 通道融合 (Channel Fusion):
           OR运算融合二值化和边缘信息：
           Result = Binary_Channel ∨ Edge_Channel
           结合区域信息和边界信息，提高检测鲁棒性。
    
    激光干扰处理策略 (Laser Interference Handling Strategy):
        问题：激光照射在黑色边框上会产生亮斑，导致边框断裂
        解决：使用30×30大尺寸闭运算核，有效桥接激光造成的间隙
        原理：大核能够跨越较大的断裂区域，重建完整边界
        
    参数选择依据 (Parameter Selection Basis):
        - 二值化阈值：基于目标与背景的对比度差异
        - 闭运算核大小：根据激光光斑的典型尺寸确定
        - Canny阈值：平衡边缘敏感性和噪声抑制
        
    适用场景 (Application Scenarios):
        - 激光追踪系统中的目标预处理
        - 强光干扰环境下的图像处理
        - 黑色目标在复杂背景中的分离
        
    Returns:
        numpy.ndarray: 预处理后的二值图像
                      白色区域(255)：目标和边缘
                      黑色区域(0)：背景
    """
    # 步骤1: BGR → 灰度图转换
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 降维处理，便于后续操作
    
    # 步骤2: 反向二值化 - 黑色目标变白色前景
    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    # 阈值35: 低于35的像素→255(白), 高于35的像素→0(黑)
    
    # 步骤3: 形态学闭运算 - 修复激光干扰断裂
    kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))  # 30×30矩形核
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_closing)
    # 闭运算: 先膨胀(连接断点) → 后腐蚀(恢复尺寸)
    
    # 步骤4: Canny边缘检测 - 提取边界特征
    edges = cv2.Canny(gray, CANNY_LOWER, CANNY_UPPER, apertureSize=3)
    # 双阈值: 50(弱边缘) + 150(强边缘), 3×3 Sobel算子
    
    # 步骤5: 融合两个通道 - 区域+边缘信息
    combined = cv2.bitwise_or(binary_closed, edges)  # 逻辑或: 任一通道有信号就保留
    
    return combined  # 返回融合后的二值图像

def calculate_angle(p1, p2, p3):
    """计算三点构成的角度（以p2为顶点）"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    
    if len1 == 0 or len2 == 0:
        return 0
    
    cos_angle = np.dot(v1, v2) / (len1 * len2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def calculate_side_lengths(corners):
    """计算四边形的四条边长"""
    sides = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        sides.append(length)
    return sides

def check_rectangle_geometry(corners, angle_tolerance=None, side_ratio_tolerance=None):
    """矩形几何检查"""
    if angle_tolerance is None:
        angle_tolerance = ANGLE_TOLERANCE
    if side_ratio_tolerance is None:
        side_ratio_tolerance = SIDE_RATIO_TOLERANCE
        
    if len(corners) != 4:
        return False
    
    angles = []
    for i in range(4):
        curr = corners[i]
        prev = corners[(i-1) % 4]
        next = corners[(i+1) % 4]
        angle = calculate_angle(prev, curr, next)
        angles.append(angle)
    
    valid_angles = [abs(angle - 90) <= angle_tolerance for angle in angles]
    angle_check = all(valid_angles)
    
    sides = calculate_side_lengths(corners)
    side1_ratio = abs(sides[0] - sides[2]) / max(sides[0], sides[2]) if max(sides[0], sides[2]) > 0 else 1
    side2_ratio = abs(sides[1] - sides[3]) / max(sides[1], sides[3]) if max(sides[1], sides[3]) > 0 else 1
    side_check = side1_ratio <= side_ratio_tolerance and side2_ratio <= side_ratio_tolerance
    
    min_side = min(sides)
    max_side = max(sides)
    side_aspect_ratio = max_side / min_side if min_side > 0 else float('inf')
    reasonable_shape = side_aspect_ratio <= 10 and min_side >= 20
    
    return angle_check and side_check and reasonable_shape

def optimize_quadrilateral(contour):
    """对轮廓进行外接图形优化"""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(contour, True)
    
    best_approx = None
    best_score = 0
    
    for epsilon_factor in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
        epsilon = epsilon_factor * perimeter
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) == 4:
            approx_area = cv2.contourArea(approx)
            original_area = cv2.contourArea(contour)
            
            if original_area > 0:
                area_ratio = approx_area / original_area
                if 0.8 <= area_ratio <= 1.2:
                    score = 1.0 - abs(1.0 - area_ratio)
                    if score > best_score:
                        best_score = score
                        best_approx = approx
    
    if best_approx is None:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        best_approx = np.int32(box).reshape(-1, 1, 2)
    
    return best_approx

def sort_corners(corners):
    """
    智能角点排序算法 (Intelligent Corner Point Sorting Algorithm)
    
    算法目标 (Algorithm Objective):
        将任意顺序的四个角点按照标准顺序重新排列：左上 → 右上 → 右下 → 左下
        确保透视变换和几何分析的准确性。
        
    算法原理 (Algorithm Principle):
        采用基于极坐标的角度排序方法，结合几何重心计算和象限分析：
        
        1. 重心计算 (Centroid Calculation):
           Centroid = (Σx_i/4, Σy_i/4)
           计算四个角点的几何中心作为极坐标原点
           
        2. 极角计算 (Polar Angle Calculation):  
           θ_i = arctan2(y_i - centroid_y, x_i - centroid_x)
           计算每个角点相对于重心的极角
           
        3. 角度归一化 (Angle Normalization):
           θ_normalized = (θ + 2π) mod 2π ∈ [0, 2π)
           将角度统一到[0, 2π)区间，避免负角度
           
        4. 起始点定位 (Starting Point Location):
           寻找最接近225°(5π/4)方向的角点作为左上角起点
           该角度对应第三象限，符合图像坐标系的左上角位置
           
        5. 顺时针排序 (Clockwise Ordering):
           从起始点开始按顺时针方向排列其余角点
           
        6. 几何验证 (Geometric Verification):
           基于Y坐标分组验证和X坐标排序进行二次校正
           确保结果的几何合理性
           
    排序策略 (Sorting Strategy):
        主策略：基于极角的顺时针排序
        备用策略：基于象限分组的坐标排序
        验证机制：Y坐标分组 + X坐标排序校验
        
    坐标系说明 (Coordinate System):
        图像坐标系：原点(0,0)在左上角，X轴向右，Y轴向下
        角点顺序：左上(TL) → 右上(TR) → 右下(BR) → 左下(BL)
        
    鲁棒性设计 (Robustness Design):
        - 处理任意初始角点顺序
        - 适应不同形状的四边形
        - 容忍一定程度的几何畸变
        - 提供几何校验机制
        
    Args:
        corners (numpy.ndarray): 待排序的四个角点
                               形状: (4, 2) 或 (4, 1, 2)
                               格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                               
    Returns:
        numpy.ndarray: 排序后的角点数组 (4, 2)
                      顺序: [左上, 右上, 右下, 左下]
                      
    算法复杂度 (Algorithm Complexity):
        时间复杂度: O(1) - 固定4个点的处理
        空间复杂度: O(1) - 固定大小的数据结构
        
    应用实例 (Application Example):
        >>> corners = np.array([[100,300], [300,100], [400,400], [200,350]])
        >>> sorted_corners = sort_corners(corners)
        >>> print(sorted_corners)
        [[300 100]  # 右上 → 左上 (重新排序后)
         [100 300]  # 左下 → 右上  
         [400 400]  # 右下 → 右下
         [200 350]] # ? → 左下
    """
    pts = corners.reshape(4, 2)
    centroid = np.mean(pts, axis=0)
    
    angles = []
    for pt in pts:
        angle = np.arctan2(pt[1] - centroid[1], pt[0] - centroid[0])
        angles.append(angle)
    
    angles = [(a + 2*np.pi) % (2*np.pi) for a in angles]
    sorted_indices = np.argsort(angles)
    
    target_angle = 5*np.pi/4
    best_start_idx = 0
    min_diff = float('inf')
    
    for i, idx in enumerate(sorted_indices):
        angle_diff = abs(angles[idx] - target_angle)
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        if angle_diff < min_diff:
            min_diff = angle_diff
            best_start_idx = i
    
    ordered_corners = []
    for i in range(4):
        idx = sorted_indices[(best_start_idx + i) % 4]
        ordered_corners.append(pts[idx])
    
    ordered_corners = np.array(ordered_corners)
    y_coords = ordered_corners[:, 1]
    y_mean = np.mean(y_coords)
    
    top_mask = y_coords <= y_mean
    bottom_mask = y_coords > y_mean
    
    if np.sum(top_mask) >= 2 and np.sum(bottom_mask) >= 2:
        top_points = ordered_corners[top_mask]
        bottom_points = ordered_corners[bottom_mask]
        
        top_sorted = top_points[np.argsort(top_points[:, 0])]
        bottom_sorted = bottom_points[np.argsort(bottom_points[:, 0])]
        
        if len(top_sorted) == 2 and len(bottom_sorted) == 2:
            final_corners = np.array([
                top_sorted[0],      # 左上
                top_sorted[1],      # 右上  
                bottom_sorted[1],   # 右下
                bottom_sorted[0]    # 左下
            ])
        else:
            final_corners = ordered_corners
    else:
        final_corners = ordered_corners
    
    return final_corners.astype(np.int32)

def sort_corners_robust(corners):
    """改进的角点排序算法（保持向后兼容）"""
    return sort_corners(corners)

def find_rectangles(contours, hierarchy):
    """
    智能矩形检测算法 (Intelligent Rectangle Detection Algorithm)
    
    功能描述 (Function Description):
        从轮廓集合中识别并提取有效的矩形目标，采用多层过滤和智能匹配策略，
        有效应对复杂环境下的干扰项，确保检测精度和稳定性。
        
    算法流程 (Algorithm Workflow):
        1. 初步过滤：基于面积和周长的粗筛选
        2. 几何验证：四边形近似和矩形特征检查
        3. 关系分析：识别父子嵌套关系
        4. 面积比过滤：排除面积比异常的干扰项
        5. 智能配对：选择最佳矩形组合
        
    多重检测策略 (Multi-Detection Strategy):
        优先级1：基于轮廓层次的父子关系检测
        优先级2：基于几何位置的嵌套关系检测  
        优先级3：基于面积排序的备选方案
        
    抗干扰设计 (Anti-Interference Design):
        - 面积比阈值过滤：排除面积差异过大的虚假嵌套
        - 几何特征验证：确保检测对象符合矩形特征
        - 多重冗余检测：提供多种检测路径增强鲁棒性
        
    Args:
        contours (list): OpenCV轮廓列表
                        每个元素为numpy数组，表示一个轮廓的点集
        hierarchy (numpy.ndarray): 轮廓层次结构信息
                                 形状: (1, N, 4)，N为轮廓数量
                                 每行格式: [next, previous, first_child, parent]
                                 
    Returns:
        list: 检测到的有效矩形列表
             每个元素为元组: (contour, corners, area)
             - contour: 原始轮廓数据
             - corners: 排序后的四个角点坐标 (4×2数组)
             - area: 轮廓面积(像素²)
             
    质量评估指标 (Quality Assessment):
        - 角度误差：所有内角与90°的偏差
        - 边长比例：对边长度的相似性
        - 面积比：嵌套矩形的面积关系
        - 几何稳定性：轮廓形状的规则性
        
    应用场景 (Application Scenarios):
        - 双层嵌套目标检测(外框+内框)
        - 复杂背景下的矩形识别
        - 实时视觉跟踪中的目标定位
        
    性能优化 (Performance Optimization):
        - 早期过滤减少计算量
        - 层次化处理提高效率
        - 智能排序优化选择过程
    """
    # 第一阶段: 初步筛选和几何验证
    valid_rects = []  # 存储通过验证的矩形候选
    
    for i, cnt in enumerate(contours):
        # 步骤1: 计算基本几何特征
        area = cv2.contourArea(cnt)          # 轮廓面积(像素²)
        perimeter = cv2.arcLength(cnt, True) # 轮廓周长(像素)
        
        # 步骤2: 粗筛选 - 过滤过小的噪声轮廓  
        if area < 100 or perimeter < 20:  # 面积<100px² 或 周长<20px
            continue  # 跳过太小的轮廓
            
        # 步骤3: Douglas-Peucker轮廓近似为四边形
        epsilon = 0.02 * perimeter  # 近似精度 = 周长的2%
        approx = cv2.approxPolyDP(cnt, epsilon, True)  # 轮廓简化
        
        # 步骤4: 四边形检查和几何验证
        if len(approx) == 4:  # 确保简化后是四边形
            sorted_corners = approx.reshape(4, 2)  # 提取4个角点
            
            # 步骤5: 矩形几何特征验证(角度+边长比)
            if check_rectangle_geometry(sorted_corners):
                x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
                parent = hierarchy[0][i][3] if hierarchy is not None else -1  # 父轮廓索引
                
                # 通过验证，加入候选列表
                valid_rects.append((cnt, approx, area, x, y, w, h, i, parent))
    
    # 第二阶段: 智能筛选和配对
    if len(valid_rects) == 0:  # 没有找到有效矩形
        return []  # 返回空列表
    
    valid_rects.sort(key=lambda x: x[2])  # 按面积从小到大排序
    selected_rects = []  # 最终选择的矩形对
    
    # 策略1: 基于轮廓层次结构的父子关系检测
    parent_child_pairs = []  # 存储有效的父子矩形对
    for rect in valid_rects:
        cnt, approx, area, x, y, w, h, idx, parent_idx = rect
        if parent_idx != -1:  # 有父轮廓(即嵌套关系)
            # 查找对应的父矩形
            parent_rect = None
            for parent_candidate in valid_rects:
                if parent_candidate[7] == parent_idx:  # 索引匹配
                    parent_rect = parent_candidate
                    break
            
            if parent_rect is not None:
                # 面积比验证 - 过滤面积差异过大的虚假嵌套
                child_area = area
                parent_area = parent_rect[2]
                area_ratio = child_area / parent_area if parent_area > 0 else 0
                
                if area_ratio >= MIN_AREA_RATIO:  # 面积比 ≥ 0.7
                    parent_child_pairs.append((parent_rect, rect))
                else:
                    print(f"过滤干扰项: 面积比 {area_ratio:.3f} < {MIN_AREA_RATIO}")
    
    # 选择策略分支
    if len(parent_child_pairs) > 0:
        # 策略1成功: 使用层次结构检测到的父子关系
        parent_child_pairs.sort(key=lambda x: x[1][2])  # 按子矩形面积排序
        parent_rect, child_rect = parent_child_pairs[0]  # 选择面积最小的子矩形对
        selected_rects = [child_rect, parent_rect]  # 内矩形+外矩形
    else:
        # 策略2: 基于几何位置的嵌套关系检测
        nested_rects_geo = []  # 几何嵌套矩形列表
        
        for i, (cnt1, approx1, area1, x1, y1, w1, h1, idx1, parent1) in enumerate(valid_rects):
            valid_nesting_partners = []  # 当前矩形的有效嵌套伙伴
            
            # 与所有其他矩形比较
            for j, (cnt2, approx2, area2, x2, y2, w2, h2, idx2, parent2) in enumerate(valid_rects):
                if i == j:  # 跳过自己
                    continue
                
                margin = 10  # 嵌套判断的边界容差
                # 几何嵌套判断: 矩形1完全包含在矩形2内
                if (x1 >= x2 - margin and y1 >= y2 - margin and 
                    x1 + w1 <= x2 + w2 + margin and y1 + h1 <= y2 + h2 + margin and
                    area1 < area2 * 0.9):  # 且面积明显更小
                    
                    # 面积比过滤
                    area_ratio = area1 / area2 if area2 > 0 else 0
                    if area_ratio >= MIN_AREA_RATIO:
                        valid_nesting_partners.append((cnt2, approx2, area2))
                    else:
                        print(f"过滤几何嵌套干扰项: 面积比 {area_ratio:.3f} < {MIN_AREA_RATIO}")
            
            # 记录嵌套层级
            if len(valid_nesting_partners) > 0:
                nesting_level = len(valid_nesting_partners)  # 嵌套深度
                nested_rects_geo.append((cnt1, approx1, area1, nesting_level))
        
        # 根据嵌套情况选择矩形
        if len(nested_rects_geo) >= 2:
            # 情况2A: 多个嵌套矩形，选择嵌套最深的两个
            nested_rects_geo.sort(key=lambda x: (-x[3], x[2]))  # 按嵌套深度降序，面积升序
            selected_rects = [nested_rects_geo[0][:3], nested_rects_geo[1][:3]]
        elif len(nested_rects_geo) >= 1:
            # 情况2B: 只有一个嵌套矩形，配对一个非嵌套矩形
            selected_rects = [nested_rects_geo[0][:3]]  # 添加嵌套矩形
            
            # 寻找非嵌套矩形配对
            if len(valid_rects) > 1:
                for rect in valid_rects:
                    if not any(rect[0] is nested[0] for nested in nested_rects_geo):
                        selected_rects.append(rect[:3])
                        break
                if len(selected_rects) == 1:  # 如果没找到合适配对，使用第二大面积
                    selected_rects.append(valid_rects[1][:3])
        else:
            # 策略3: 没有明显嵌套关系，选择面积最小和最大的
            if len(valid_rects) >= 2:
                selected_rects = [valid_rects[0][:3], valid_rects[-1][:3]]  # 最小+最大面积
            elif len(valid_rects) == 1:
                selected_rects = [valid_rects[0][:3]]  # 只有一个矩形
    
    # 第三阶段: 角点排序和结果封装
    final_rects = []  # 最终结果列表
    for rect_data in selected_rects:
        cnt = rect_data[0]      # 原始轮廓
        approx = rect_data[1]   # 四边形近似角点
        area = rect_data[2]     # 轮廓面积
        
        # 角点标准化排序: 左上→右上→右下→左下
        sorted_corners = sort_corners(approx)
        final_rects.append((cnt, sorted_corners, area))
    
    return final_rects  # 返回检测到的矩形列表

def create_perspective_transform(src_corners, target_width_mm=297, target_height_mm=210, pixels_per_mm=2):
    """
    创建透视变换矩阵系统 (Create Perspective Transform Matrix System)
    
    功能描述 (Function Description):
        基于检测到的四边形角点创建透视变换矩阵，将倾斜视角的目标校正为标准正视图。
        同时计算正向和逆向变换矩阵，支持原图与校正图之间的双向坐标转换。
        
    透视变换原理 (Perspective Transform Principle):
        透视变换是一种保持直线性的非线性变换，能够校正由于相机视角、目标姿态
        造成的几何畸变。变换矩阵H为3×3齐次矩阵：
        
        [x']   [h11 h12 h13] [x]
        [y'] = [h21 h22 h23] [y]
        [w']   [h31 h32 h33] [w]
        
        实际坐标: (x'/w', y'/w')
        
    坐标系统定义 (Coordinate System Definition):
        源图像坐标系：检测到的倾斜四边形
        目标图像坐标系：标准矩形，左上角为原点
        - 左上角：(0, 0)
        - 右上角：(target_width_px-1, 0)  
        - 右下角：(target_width_px-1, target_height_px-1)
        - 左下角：(0, target_height_px-1)
        
    像素密度设计 (Pixel Density Design):
        采用毫米到像素的固定映射比例，确保校正后图像的物理尺寸准确性：
        像素数 = 物理尺寸(mm) × 像素密度(pixels/mm)
        
    Args:
        src_corners (numpy.ndarray): 源图像中的四个角点坐标
                                   形状: (4, 2) 或 (4, 1, 2)
                                   顺序: 需要是有序的四边形角点
                                   
        target_width_mm (int, optional): 目标图像宽度(毫米). 默认297 (A4纸宽度)
        target_height_mm (int, optional): 目标图像高度(毫米). 默认210 (A4纸高度)  
        pixels_per_mm (float, optional): 像素密度(像素/毫米). 默认2.0
        
    Returns:
        tuple: (M, M_inv, corrected_size)
            M (numpy.ndarray): 正向透视变换矩阵 (3×3)
                             用于将原图坐标变换到校正图坐标
            M_inv (numpy.ndarray): 逆向透视变换矩阵 (3×3)
                                 用于将校正图坐标变换回原图坐标  
            corrected_size (tuple): 校正图像尺寸 (width, height)
            
    变换矩阵应用 (Transform Matrix Usage):
        正向变换：原图 → 校正图
        >>> corrected_image = cv2.warpPerspective(src_image, M, corrected_size)
        
        逆向变换：校正图 → 原图  
        >>> original_point = cv2.perspectiveTransform(corrected_point.reshape(1,1,2), M_inv)
        
    精度考虑 (Precision Considerations):
        - 角点检测精度：影响变换矩阵准确性
        - 像素密度选择：影响校正图像分辨率和计算复杂度
        - 数值稳定性：避免奇异矩阵和数值溢出
        
    应用实例 (Application Example):
        >>> corners = np.array([[100,100], [500,120], [480,400], [80,380]])
        >>> M, M_inv, size = create_perspective_transform(corners)
        >>> print(f"校正图像尺寸: {size}")
        校正图像尺寸: (594, 420)
    """
    # 步骤1: 计算校正图像的像素尺寸
    target_width_px = int(target_width_mm * pixels_per_mm)    # 297mm×2 = 594像素
    target_height_px = int(target_height_mm * pixels_per_mm)  # 210mm×2 = 420像素
    
    # 步骤2: 定义标准矩形角点(校正后的理想坐标)
    dst_corners = np.array([
        [0, 0],                                        # 左上角(0,0)
        [target_width_px - 1, 0],                      # 右上角(593,0)
        [target_width_px - 1, target_height_px - 1],   # 右下角(593,419)
        [0, target_height_px - 1]                      # 左下角(0,419)
    ], dtype=np.float32)  # 标准矩形: 左上→右上→右下→左下
    
    # 步骤3: 源图角点排序 - 确保与目标角点一一对应
    sorted_corners = sort_corners_robust(src_corners)  # 调用排序函数统一顺序
    src_corners_sorted = np.array(sorted_corners, dtype=np.float32)
    
    # 步骤4: 计算正向透视变换矩阵 H (3×3)
    M = cv2.getPerspectiveTransform(src_corners_sorted, dst_corners)
    # 原图倾斜四边形 → 校正后标准矩形
    
    # 步骤5: 计算逆向透视变换矩阵 H^(-1)
    M_inv = cv2.getPerspectiveTransform(dst_corners, src_corners_sorted)  
    # 校正后标准矩形 → 原图倾斜四边形
    
    return M, M_inv, (target_width_px, target_height_px)  # 返回变换矩阵对+图像尺寸

def draw_target_circle(image, center, radius_mm=60, pixels_per_mm=2):
    """在图像中心绘制目标圆"""
    radius_px = int(radius_mm * pixels_per_mm)
    
    cv2.circle(image, center, radius_px, (255, 0, 255), 4)
    
    label = f"Target Circle (R={radius_mm}mm)"
    label_pos = (center[0] - 80, center[1] - radius_px - 15)
    cv2.putText(image, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    cross_size = 10
    cv2.line(image, (center[0] - cross_size, center[1]), 
             (center[0] + cross_size, center[1]), (255, 0, 255), 2)
    cv2.line(image, (center[0], center[1] - cross_size), 
             (center[0], center[1] + cross_size), (255, 0, 255), 2)
    
    return radius_px

def find_center_in_corrected_image(corrected_img, corrected_sz):
    """在校正后的图像中寻找内部矩形的中心"""
    combined = preprocess_image(corrected_img)
    if combined is None:
        return (corrected_sz[0] // 2, corrected_sz[1] // 2), None
    
    contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    selected_rects = find_rectangles(contours, hierarchy)
    
    if len(selected_rects) >= 1:
        # 使用第一个矩形（通常是内部矩形）
        cnt = selected_rects[0][0]
        corners = selected_rects[0][1]  # 排序后的角点
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return (center_x, center_y), corners
    
    center_x = corrected_sz[0] // 2
    center_y = corrected_sz[1] // 2
    return (center_x, center_y), None

def transform_point_back(point, M_inv):
    """将校正图像中的点变换回原图"""
    point_homogeneous = np.array([[point[0], point[1], 1]], dtype=np.float32).T
    transformed = M_inv @ point_homogeneous
    x = transformed[0, 0] / transformed[2, 0]
    y = transformed[1, 0] / transformed[2, 0]
    return (int(x), int(y))

def transform_point_to_corrected(point, M_matrix):
    """将原图中的点变换到校正图像"""
    point_homogeneous = np.array([[point[0], point[1], 1]], dtype=np.float32).T
    transformed = M_matrix @ point_homogeneous
    x = transformed[0, 0] / transformed[2, 0]
    y = transformed[1, 0] / transformed[2, 0]
    return (int(x), int(y))

def calculate_error_in_mm(point1, point2, pixels_per_mm=2):
    """计算两点间的误差（毫米）"""
    dx_pixels = point2[0] - point1[0]
    dy_pixels = point2[1] - point1[1]
    dx_mm = dx_pixels / pixels_per_mm
    dy_mm = dy_pixels / pixels_per_mm
    return dx_mm, dy_mm

def check_point_in_circle(point, center, radius):
    """检查点是否在圆内"""
    distance = np.sqrt((point[0] - center[0])**2 + (point[1] - center[1])**2)
    return distance <= radius

def stop_capture():
    """停止采集"""
    global connected
    if connected and cam:
        cam.MV_CC_StopGrabbing()
        print("✓ 停止采集")

def disconnect_camera():
    """断开连接"""
    global connected, cam, serial_port
    if connected and cam:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        connected = False
        print("✓ 设备断开")
    
    if serial_port:
        serial_port.close()
        print("✓ 串口关闭")

def run_vision_system():
    """
    电赛激光追踪小车主控制系统 (Main Control System for Laser Tracking Car)
    
    系统架构 (System Architecture):
        这是整个激光追踪小车视觉导航系统的核心控制函数，集成了硬件控制、
        图像处理、目标检测、导航计算等所有关键功能模块。
        
    运行流程 (Execution Flow):
        1. 系统初始化阶段 (System Initialization):
           ├── 串口通信初始化
           ├── 相机设备枚举和连接  
           ├── 相机参数配置(曝光、增益、帧率)
           └── 图像采集启动
           
        2. 实时处理循环 (Real-time Processing Loop):
           ├── 图像采集 → 预处理 → 目标检测
           ├── 透视校正 → 距离估算 → 激光点映射
           ├── 误差计算 → 串口数据传输
           └── 可视化显示 → 用户交互处理
           
        3. 系统退出清理 (System Exit Cleanup):
           ├── 停止图像采集
           ├── 关闭相机连接
           ├── 关闭串口通信
           └── 释放OpenCV资源
    
    核心算法集成 (Core Algorithm Integration):
        - 多模态图像预处理: 二值化 + 边缘检测 + 形态学处理
        - 智能矩形检测: 轮廓分析 + 几何验证 + 嵌套关系识别
        - 透视变换校正: 角点排序 + 透视矩阵计算 + 图像校正
        - 基于面积的距离估算: 面积测量 + 反比例计算
        - 距离映射激光导航: 线性坐标映射 + 误差计算
        
    实时性能保障 (Real-time Performance Assurance):
        - 图像处理优化: OpenCV多线程 + SIMD指令集
        - 内存管理: 缓存重用 + 避免频繁内存分配
        - 计算负载均衡: 关键路径优化 + 非关键功能可选
        - 硬件加速: 利用Jetson GPU加速(如果可用)
        
    调试和监控功能 (Debug and Monitoring Features):
        - 实时参数调整: 支持运行时修改检测阈值
        - 多窗口可视化: 原图、校正图、中间处理结果
        - 性能监控: FPS计算、处理时间统计
        - HSV颜色分析: 交互式颜色空间调试
        - 参数状态显示: 实时显示所有关键参数
        
    通信协议 (Communication Protocol):
        - 自定义二进制协议: 高效率数据传输
        - 多类型数据支持: 坐标、误差、距离、状态
        - 实时性保障: 最小化通信延迟
        - 错误恢复机制: 通信异常处理
        
    用户交互界面 (User Interface):
        - 键盘快捷键: 参数调整、功能切换、系统控制
        - 鼠标交互: 点击获取HSV信息、坐标标定
        - 状态显示: 系统状态、检测结果、性能指标
        - 调试信息: 实时参数、检测过程、错误提示
        
    系统容错设计 (Fault Tolerance Design):
        - 硬件异常处理: 相机断开、串口异常恢复
        - 算法鲁棒性: 检测失败处理、异常数据过滤
        - 资源保护: 内存泄露防护、文件句柄管理
        - 优雅退出: 完整的资源清理机制
        
    应用场景适配 (Application Scenario Adaptation):
        - 电子设计竞赛: 标准化目标检测和追踪
        - 工业自动化: 高精度定位和导航
        - 教学演示: 可视化调试和参数学习
        - 研究开发: 算法验证和性能测试
    """
    global clicked_point, final_center_point, corrected_image, M, M_inv, corrected_size, center_in_corrected
    global laser_point, laser_enabled, current_frame_for_hsv, hsv_output_requested
    global DEBUG_SHOW_BINARY, DEBUG_SHOW_EDGES, DEBUG_SHOW_COMBINED
    global BINARY_THRESHOLD, BINARY_THRESHOLD_CORRECTED, MIN_AREA_RATIO, windows_created

    
    print("=== Jetson Orin NX 海康相机视觉检测系统 + 距离映射激光点 ===")
    
    if not SDK_OK:
        print("SDK未正确加载，无法运行")
        return
    
    # 设置串口
    print("\n步骤1: 串口配置")
    if not setup_serial():
        print("⚠️ 串口配置失败，将在无串口模式下运行")
    
    # 查找设备
    print("\n步骤2: 搜索相机设备")
    devices = list_devices()
    if not devices:
        print("未找到设备")
        return
    
    # 连接第一个设备
    print("\n步骤3: 连接相机")
    if not connect_camera(0):
        print("设备连接失败")
        return
    
    # 设置相机参数
    print("\n步骤4: 配置相机参数")
    set_exposure(5000)
    set_gain(12)
    set_framerate(100)
    
    # 开始采集
    print("\n步骤5: 启动图像采集")
    if not start_capture():
        print("启动采集失败")
        return
    
    # 设置OpenCV优化
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    
    # 设置鼠标回调
    cv2.namedWindow("Jetson Hik Camera Vision", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Jetson Hik Camera Vision", mouse_callback)
    
    print("\n✓ 视觉检测系统运行中...")
    print(f"串口状态: {'已连接' if serial_enabled else '未连接'}")
    if serial_enabled:
        print(f"串口配置: {selected_port} @ {selected_baudrate}")
    print("操作: Q-退出 S-保存 R-重置 L-显示激光映射参数")
    print("调试: B-二值化 E-边缘 C-合并 D-全部调试 1-打印参数")
    print("阈值: +/-主阈值 [/]校正阈值 A/Z面积比阈值")
    print("💡 点击图像任意位置可获取该点的HSV颜色信息")
    print("💡 激光点坐标基于距离自动映射：500mm-1500mm 对应 Y坐标 240-250")
    print("-" * 50)
    
    # ========== 主处理循环 ==========
    while True:
        frame_start = time.time()  # 记录帧处理开始时间

        # --- 图像采集阶段 ---
        frame_main = get_image()  # 从相机获取一帧图像
        if frame_main is None:    # 获取失败时跳过本帧
            continue
            
        # --- 图像预处理阶段 ---
        frame_main = cv2.resize(frame_main, (640, 480))  # 统一图像尺寸
        current_frame_for_hsv = frame_main.copy()        # 缓存HSV分析用图像

        # --- 图像处理管道 ---
        # 步骤1: 高斯模糊降噪
        blurred = cv2.GaussianBlur(frame_main, (GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIZE), 0)
        # 步骤2: 多模态预处理(二值化+边缘检测+形态学)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, BINARY_THRESHOLD_CORRECTED, 255, cv2.THRESH_BINARY_INV)
        
        # 步骤3: 形态学闭运算修复激光干扰
        kernel_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_closing)
        
        # 步骤4: 边缘检测和信息融合
        edges = cv2.Canny(gray, CANNY_LOWER, CANNY_UPPER, apertureSize=3)
        combined = cv2.bitwise_or(binary_closed, edges)  # 融合区域+边缘信息
        
        if combined is None:  # 处理失败时跳过
            continue
        
        # --- 目标检测阶段 ---  
        contours, hierarchy = cv2.findContours(combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        selected_rects = find_rectangles(contours, hierarchy)  # 智能矩形检测
        
        # --- 初始化检测结果变量 ---
        detection_status = "未检测到矩形"     # 检测状态文本
        laser_pt = None                     # 激光点坐标 
        laser_status = "激光点: 无距离数据"   # 激光状态文本
        
        # --- 目标检测成功分支 ---
        if len(selected_rects) >= 1:
            detection_status = f"检测到 {len(selected_rects)} 个矩形"
            
            # 选择外框矩形进行处理(面积最大的通常是外框)
            outer_rect_corners = selected_rects[-1][1]  # 外框角点(已排序)
            outer_rect_area = selected_rects[-1][2]     # 外框面积
            
            # 距离估算: 面积→距离转换
            estimated_distance = estimate_distance_by_area(outer_rect_area)
            
            # 激光点映射: 距离→坐标转换  
            laser_pt = calculate_laser_position_by_distance(estimated_distance)
            laser_status = f"激光点: 映射 ({laser_pt[0]}, {laser_pt[1]}) | {estimated_distance:.1f}mm"
            
            # 打印距离估计信息到命令行
            #print_calibration_info(outer_rect_area, estimated_distance)
            
            # 创建透视变换
            M, M_inv, corrected_size = create_perspective_transform(
                outer_rect_corners.reshape(4, 2), 
                target_width_mm=297,
                target_height_mm=210,
                pixels_per_mm=1.5
            )
            
            # 应用透视变换
            corrected_image = cv2.warpPerspective(frame_main, M, corrected_size)
            
            # 在校正后的图像中寻找中心
            center_in_corrected, inner_corners = find_center_in_corrected_image(
                corrected_image, corrected_size
            )
            
            # 将中心点变换回原图
            final_center_point = transform_point_back(center_in_corrected, M_inv)
            
            # 在校正图像中绘制检测结果
            if inner_corners is not None:
                cv2.drawContours(corrected_image, [inner_corners.reshape(-1, 1, 2)], -1, (0, 255, 0), 2)
            
            # 标记中心点
            cv2.circle(corrected_image, center_in_corrected, 8, (0, 0, 255), -1)
            
            # 绘制目标圆
            target_radius_px = draw_target_circle(
                corrected_image, center_in_corrected, 
                radius_mm=60, pixels_per_mm=1.5
            )
            
            # 处理激光点在校正图像中的显示 - 使用基于距离映射的坐标
            if laser_pt is not None and M is not None:
                # 将映射的激光点坐标转换到校正图像坐标系
                laser_pt_corrected = transform_point_to_corrected(laser_pt, M)
                
                # 检查点是否在目标圆内
                in_target_circle = check_point_in_circle(
                    laser_pt_corrected, center_in_corrected, target_radius_px
                )
                
                # 设置颜色
                point_color = (0, 255, 0) if in_target_circle else (0, 0, 255)
                
                # 绘制激光点
                cv2.circle(corrected_image, laser_pt_corrected, 8, point_color, 2)
                cv2.circle(corrected_image, laser_pt_corrected, 3, (0, 255, 255), -1)
                
                # 计算误差
                dx_mm, dy_mm, distance_mm = calculate_laser_error(
                    laser_pt_corrected, center_in_corrected, pixels_per_mm=1.5
                )
                
                if dx_mm is not None:
                    # 通过串口发送误差数据
                    if serial_enabled:
                        send_error_data(dx_mm, dy_mm, estimated_distance, data_type="laser")
                    
                    # 显示误差信息
                    info_y = 200
                    cv2.putText(corrected_image, f'Laser: {laser_pt_corrected}', 
                               (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1)
                    cv2.putText(corrected_image, f'Laser X: {dx_mm:.1f}mm', 
                               (10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(corrected_image, f'Laser Y: {dy_mm:.1f}mm', 
                               (10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.putText(corrected_image, f'Laser Dist: {distance_mm:.1f}mm', 
                               (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # 显示点状态
                    status_text = "✓ LASER IN TARGET" if in_target_circle else "✗ LASER OUTSIDE"
                    status_color = (0, 255, 0) if in_target_circle else (0, 0, 255)
                    cv2.putText(corrected_image, status_text, (10, info_y + 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    
                    # 显示当前使用的点类型
                    mode_text = "Mode: DISTANCE MAPPED"
                    mode_color = (0, 255, 0)
                    cv2.putText(corrected_image, mode_text, (10, info_y + 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 2)
                    
                    # 绘制误差线
                    cv2.line(corrected_image, center_in_corrected, 
                            laser_pt_corrected, (0, 255, 255), 2)
            
            # 在校正图像上标注坐标系信息
            cv2.putText(corrected_image, 'Corrected View', 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(corrected_image, f'Size: {corrected_size[0]}x{corrected_size[1]}', 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 显示中心点坐标
            cv2.putText(corrected_image, f'Center: {center_in_corrected}', 
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            detection_status = "透视校正成功"
            
        else:
            final_center_point = (-1, -1)
            corrected_image = None
        
        # 在原图中绘制检测结果
        estimated_distance = -1  # 初始化距离变量
        for i, rect_data in enumerate(selected_rects):
            color = (0, 255, 0) if i == len(selected_rects)-1 else (255, 0, 0)
            corners = rect_data[1].reshape(-1, 1, 2)
            cv2.drawContours(frame_main, [corners], -1, color, 2)
            
            # 如果是最大的矩形（用于透视校正的矩形），显示距离信息
            if i == len(selected_rects)-1:
                rect_area = rect_data[2]
                estimated_distance = estimate_distance_by_area(rect_area)
            
            # 标注矩形序号
            rect_center = np.mean(corners.reshape(4, 2), axis=0).astype(int)
            cv2.putText(frame_main, f'R{i+1}', tuple(rect_center), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 绘制最终中心点
        if final_center_point != (-1, -1):
            cv2.circle(frame_main, final_center_point, 8, (0, 0, 255), -1)
            cv2.putText(frame_main, f'Center: {final_center_point}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 绘制点击点
        if clicked_point is not None:
            cv2.circle(frame_main, clicked_point, 6, (255, 0, 0), -1)
            cv2.putText(frame_main, f'Click: {clicked_point}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 处理HSV信息输出请求
            if hsv_output_requested:
                print(f"正在获取坐标 {clicked_point} 的HSV信息...")
                output_hsv_info(current_frame_for_hsv, clicked_point[0], clicked_point[1])
                hsv_output_requested = False  # 重置标志
        
        # 绘制激光点（基于距离映射）
        if laser_pt is not None:
            draw_laser_position_info(frame_main, laser_pt, estimated_distance)
        
        # 性能信息
        fps = get_fps()
        frame_time = (time.time() - frame_start) * 1000
        
        # 显示系统信息
        info_y_start = 90
        cv2.putText(frame_main, f'FPS: {fps:.1f} | Frame: {frame_time:.1f}ms', 
                   (10, info_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 串口状态显示
        serial_status = f"Serial: {selected_port}@{selected_baudrate}" if serial_enabled else "Serial: OFF"
        cv2.putText(frame_main, serial_status, 
                   (10, info_y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                   (0, 255, 0) if serial_enabled else (0, 0, 255), 1)
        
        cv2.putText(frame_main, f'Status: {detection_status}', 
                   (10, info_y_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示距离估计信息
        if estimated_distance > 0:
            distance_text = f'Distance: {estimated_distance:.1f}mm ({estimated_distance/10:.1f}cm)'
            cv2.putText(frame_main, distance_text, 
                       (10, info_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame_main, f'{laser_status}', 
                       (10, info_y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0), 1)
            cv2.putText(frame_main, f'Res: {frame_main.shape[1]}x{frame_main.shape[0]}', 
                       (10, info_y_start + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame_main, f'{laser_status}', 
                       (10, info_y_start + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (128, 128, 128), 1)
            cv2.putText(frame_main, f'Res: {frame_main.shape[1]}x{frame_main.shape[0]}', 
                       (10, info_y_start + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加操作提示 - 动态调整位置
        hint_y_base = frame_main.shape[0] - 70
        cv2.putText(frame_main, 'Q:Quit S:Save R:Reset L:LaserInfo B:Binary E:Edges C:Combined D:Debug 1:PrintParams', 
                   (10, hint_y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
        cv2.putText(frame_main, '+/-:MainThresh [/]:CorrectedThresh', 
                   (10, hint_y_base + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (128, 255, 255), 1)
        cv2.putText(frame_main, f'Debug: Binary:{DEBUG_SHOW_BINARY} Edges:{DEBUG_SHOW_EDGES} Combined:{DEBUG_SHOW_COMBINED}', 
                   (10, hint_y_base + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 255, 128), 1)
        cv2.putText(frame_main, f'Thresholds: Main:{BINARY_THRESHOLD} Corrected:{BINARY_THRESHOLD_CORRECTED}', 
                   (10, hint_y_base + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 128, 255), 1)
        
        # 显示窗口
        cv2.imshow("Jetson Hik Camera Vision", frame_main)
        
        # 调试窗口显示 - 根据调试开关控制
        if DEBUG_SHOW_BINARY:
            safe_show_window("binary", binary)
        else:
            # 如果关闭了调试，确保关闭窗口
            safe_destroy_window("binary")
        
        if DEBUG_SHOW_EDGES:
            safe_show_window("edges", edges)
        else:
            safe_destroy_window("edges")
                
        if DEBUG_SHOW_COMBINED:
            safe_show_window("combined", combined)
        else:
            safe_destroy_window("combined")
        
        # 显示校正图像
        if corrected_image is not None:
            display_height = min(400, corrected_image.shape[0])
            if corrected_image.shape[0] > display_height:
                display_width = int(display_height * corrected_image.shape[1] / corrected_image.shape[0])
                display_corrected = cv2.resize(corrected_image, (display_width, display_height))
            else:
                display_corrected = corrected_image
            cv2.imshow("Perspective Corrected", display_corrected)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("用户退出")
            break
        elif key == ord('s') and frame_main is not None:
            # 保存图像
            timestamp = int(time.time())
            save_dir = '/home/nvidia' if os.path.exists('/home/nvidia') else '/tmp'
            
            original_filename = f'{save_dir}/jetson_hik_original_{timestamp}.jpg'
            cv2.imwrite(original_filename, frame_main)
            print(f"✓ 保存原图: {original_filename}")
            
            if corrected_image is not None:
                corrected_filename = f'{save_dir}/jetson_hik_corrected_{timestamp}.jpg'
                cv2.imwrite(corrected_filename, corrected_image)
                print(f"✓ 保存校正图: {corrected_filename}")
                
        elif key == ord('r'):
            # 重置点击点
            clicked_point = None
            print("✓ 重置点击点")
            
        elif key == ord('l'):
            # 激光点映射系统总是启用的，这里可以显示当前映射参数
            print(f"✓ 激光映射参数: 距离范围 {MIN_DISTANCE_MM}-{MAX_DISTANCE_MM}mm")
            print(f"  X坐标固定: {LASER_X_FIXED}")
            print(f"  Y坐标范围: {LASER_Y_MIN_DISTANCE}-{LASER_Y_MAX_DISTANCE}")
                
        elif key == ord('b'):
            # 切换二值化调试显示
            DEBUG_SHOW_BINARY = not DEBUG_SHOW_BINARY
            print(f"✓ 二值化显示: {'开启' if DEBUG_SHOW_BINARY else '关闭'}")
            
        elif key == ord('e'):
            # 切换边缘检测调试显示
            DEBUG_SHOW_EDGES = not DEBUG_SHOW_EDGES
            print(f"✓ 边缘检测显示: {'开启' if DEBUG_SHOW_EDGES else '关闭'}")
            
        elif key == ord('c'):
            # 切换合并结果调试显示
            DEBUG_SHOW_COMBINED = not DEBUG_SHOW_COMBINED
            print(f"✓ 合并结果显示: {'开启' if DEBUG_SHOW_COMBINED else '关闭'}")
            
        elif key == ord('d'):
            # 切换所有调试显示
            debug_all = not (DEBUG_SHOW_BINARY or DEBUG_SHOW_EDGES or DEBUG_SHOW_COMBINED)
            DEBUG_SHOW_BINARY = debug_all
            DEBUG_SHOW_EDGES = debug_all
            DEBUG_SHOW_COMBINED = debug_all
            print(f"✓ 所有调试显示: {'开启' if debug_all else '关闭'}")
            
        elif key == ord('+') or key == ord('='):
            # 增加主二值化阈值
            BINARY_THRESHOLD = min(255, BINARY_THRESHOLD + 5)
            print(f"✓ 主二值化阈值: {BINARY_THRESHOLD}")
            
        elif key == ord('-'):
            # 减少主二值化阈值
            BINARY_THRESHOLD = max(0, BINARY_THRESHOLD - 5)
            print(f"✓ 主二值化阈值: {BINARY_THRESHOLD}")
            
        elif key == ord('['):
            # 减少校正图像二值化阈值
            BINARY_THRESHOLD_CORRECTED = max(0, BINARY_THRESHOLD_CORRECTED - 5)
            print(f"✓ 校正图像二值化阈值: {BINARY_THRESHOLD_CORRECTED}")
            
        elif key == ord(']'):
            # 增加校正图像二值化阈值
            BINARY_THRESHOLD_CORRECTED = min(255, BINARY_THRESHOLD_CORRECTED + 5)
            print(f"✓ 校正图像二值化阈值: {BINARY_THRESHOLD_CORRECTED}")
            
        elif key == ord('1'):
            # 打印当前所有阈值参数
            print("\n=== 当前阈值参数 ===")
            print(f"主二值化阈值: {BINARY_THRESHOLD}")
            print(f"校正图像二值化阈值: {BINARY_THRESHOLD_CORRECTED}")
            print(f"Canny下阈值: {CANNY_LOWER}")
            print(f"Canny上阈值: {CANNY_UPPER}")
            print(f"高斯模糊大小: {GAUSSIAN_BLUR_SIZE}")
            print(f"最小轮廓面积: {MIN_CONTOUR_AREA}")
            print(f"最大轮廓面积: {MAX_CONTOUR_AREA}")
            print(f"角度容差: {ANGLE_TOLERANCE}")
            print(f"边长比容差: {SIDE_RATIO_TOLERANCE}")
            print(f"最小面积比: {MIN_AREA_RATIO} (嵌套矩形过滤)")
            print("--- 距离估计参数 ---")
            print(f"参考面积: {REFERENCE_AREA_PIXELS} 像素")
            print(f"参考距离: {REFERENCE_DISTANCE_MM} mm")
            print(f"实际物体面积: {REAL_OBJECT_AREA_MM2} mm²")
            print("--- 激光点映射参数 ---")
            print(f"距离范围: {MIN_DISTANCE_MM}-{MAX_DISTANCE_MM} mm")
            print(f"X坐标固定: {LASER_X_FIXED}")
            print(f"Y坐标范围: {LASER_Y_MIN_DISTANCE}-{LASER_Y_MAX_DISTANCE}")
            print("=====================")
            
        elif key == ord('a'):
            # 增加面积比阈值
            MIN_AREA_RATIO = min(1.0, MIN_AREA_RATIO + 0.05)
            print(f"✓ 最小面积比阈值: {MIN_AREA_RATIO:.2f}")
            
        elif key == ord('z'):
            # 减少面积比阈值
            MIN_AREA_RATIO = max(0.1, MIN_AREA_RATIO - 0.05)
            print(f"✓ 最小面积比阈值: {MIN_AREA_RATIO:.2f}")

def main():
    """主函数"""
    print("=== 海康相机视觉系统 v2.3 (距离映射激光点版) ===")
    
    if not SDK_OK:
        print("❌ SDK未加载，请安装海康威视MVS SDK")
        return
    
    try:
        # 运行视觉检测系统
        run_vision_system()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        print("\n正在清理资源...")
        stop_capture()
        disconnect_camera()
        cleanup_all_debug_windows()  # 先清理调试窗口
        cv2.destroyAllWindows()      # 再清理所有窗口
        print("✓ 系统程序结束")

if __name__ == "__main__":
    main()
