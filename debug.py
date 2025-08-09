#-*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import time
from connect_uav import UPUavControl
from multiprocessing import Process
from multiprocessing.managers import BaseManager

cap = cv2.VideoCapture(0)

class LineFollower():
    def __init__(self): # 是否显示二值化窗口（默认显示）
        # 实例化进程间无人机控制实例
        BaseManager.register('UPUavControl', UPUavControl)
        manager = BaseManager()
        manager.start()
        self.airplanceApi = manager.UPUavControl()
        controllerAir = Process(name="controller_air", target=self.api_init)
        controllerAir.start()

        # 控制无人机到指定高度，并且悬停5秒
        time.sleep(1)
        self.airplanceApi.onekey_takeoff(60)
        time.sleep(5)
        print("ting x5")

        # 初始化其他变量
        self.wurenji_center = (240, 180)  # 无人机中心点（图块4-5合并中心，默认640x480分辨率）
        self.threshold_value = 60  # 二值化阈值

        # 启动视频处理流程
        self.start_video()

    def api_init(self):
        print("process start")
        # 给系统一些时间来稳定或初始化
        time.sleep(0.5)
        # 控制舵机旋转90度
        self.airplanceApi.setServoPosition(90)

    def line_mode_process(self, gray):
        valid_centers, trigger_land = self.cnt_to_centers(gray)  # 获取有效中心点列表和触发降落标志
        
        # 如果触发降落条件（检查所有六个图块）
        if trigger_land:
            print("检测到所有图块轮廓数量均大于7，准备降落!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            self.airplanceApi.setServoPosition(0)
            cap.release() 
            self.airplanceApi.setMoveAction(0, 0, 0, 0)
            time.sleep(1)  # 短暂停顿
            self.airplanceApi.land()
            return True  # 返回True表示需要退出循环
            
        line_lr_speed = 0
        line_fb_speed = 200
        line_turn_speed = 0
        # ==========计算无人机中心点到前两个有效中心点连线的有符号距离及连线角度==========
        if len(valid_centers) >= 2:
            angle, distance = self.drift_process(valid_centers)
            if abs(angle) > 10:
                line_turn_speed = 200 if angle > 0 else -200
                line_fb_speed = 200
            if abs(distance) > 70:
                line_lr_speed = 100 if distance > 0 else -100
                line_fb_speed = 200
            self.airplanceApi.setMoveAction(line_lr_speed, line_fb_speed, 0, line_turn_speed)
        elif len(valid_centers) == 1:  # ================开局只有一个点，直走；漂太歪了，紧急回正================
            jinjihuizheng_x = valid_centers[0][0] - 240
            if abs(jinjihuizheng_x) > 70:
                line_lr_speed = 150 if jinjihuizheng_x > 0 else -150
                line_fb_speed = 0
            else:
                line_lr_speed = 0
                line_fb_speed = 200
            self.airplanceApi.setMoveAction(line_lr_speed, line_fb_speed, 0, 0)
        else:
            self.airplanceApi.setMoveAction(100, 100, 0, 0)
            
        return False  # 返回False表示继续循环

    def cnt_to_centers(self, gray):
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        block_height = binary.shape[0] // 6

        valid_centers = []  # 存储所有有效中心点 (x, y)
        trigger_land = [False] * 6  # 初始化6个图块的降落标志

        for block_idx in range(6):
            y_start, y_end = block_idx * block_height, (block_idx + 1) * block_height
            roi_binary = binary[y_start:y_end, :]
            roi_inv = cv2.bitwise_not(roi_binary)
            contours, _ = cv2.findContours(roi_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour_info = []  # 存储轮廓信息的列表
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 1800:
                    continue
                    
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                    
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                # 转换到原图坐标系
                cnt[:, :, 1] += y_start
                cx_real = cx
                cy_real = cy + y_start
                
                distance_to_center = abs(cx_real - 240)
                contour_info.append({
                    'center': (cx_real, cy_real),
                    'distance': distance_to_center
                })
                
            # 如果当前图块的轮廓数量大于等于7，触发降落条件
            if len(contours) >= 7:
                trigger_land[block_idx] = True
            
            # 图块中有轮廓
            if contour_info:
                # 按距离排序，找到距离中心线最近的轮廓
                contour_info.sort(key=lambda x: x['distance'])
                best = contour_info[0]
                cx_real, cy_real = best['center']
                valid_centers.append((cx_real, cy_real))  # 添加到有效中心点列表
        
        # 检查所有六个图块是否都触发降落条件
        land = all(trigger_land)
        return valid_centers, land

    def drift_process(self, valid_centers):
        pt1 = valid_centers[0]  # 第一个有效中心点（上方）
        pt2 = valid_centers[1]  # 第二个有效中心点（下方）

        dx = pt1[0] - pt2[0]  # x方向距离
        dy = pt2[1] - pt1[1]  # y方向距离
        
        angle = int(np.arctan2(dx, dy) * 180 / np.pi)  # 计算角度（相对于垂直方向）
        
        # ==========计算直线方程 Ax + By + C = 0 ==========
        A = pt2[1] - pt1[1]
        B = pt1[0] - pt2[0]
        C = pt2[0] * pt1[1] - pt1[0] * pt2[1]
        
        x0, y0 = (240, 227)  # 无人机中心点

        denominator = np.sqrt(A**2 + B**2)
        if denominator > 0:
            distance = int((A * x0 + B * y0 + C) / denominator)
        else:
            distance = 0

        # 计算垂足坐标
        dx_vector = pt2[0] - pt1[0]
        dy_vector = pt2[1] - pt1[1]
        vector_length = dx_vector**2 + dy_vector**2
        t = ((x0 - pt1[0]) * dx_vector + (y0 - pt1[1]) * dy_vector) / vector_length
        foot_x = pt1[0] + t * dx_vector
        foot_y = pt1[1] + t * dy_vector
        foot_point = (int(foot_x), int(foot_y))

        # 垂足在无人机左右判断距离正负
        cross = foot_point[0] - x0
        distance = abs(distance) * (1 if cross > 0 else -1)

        return angle, distance
        
    def start_video(self):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            resize_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
            gray = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2GRAY)

            # 处理线模式，如果返回True表示需要退出循环
            if self.line_mode_process(gray):
                break

        # 确保摄像头被释放
        if cap.isOpened():
            cap.release()
        # 舵机归零
        self.airplanceApi.setServoPosition(0)

if __name__ == '__main__':
    line_follower = LineFollower()