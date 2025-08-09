#-*- coding: utf-8 -*-
import cv2
import os
import numpy as np

class VideoDebugOp():
    def __init__(self, 
                 video_path, 
                 target_fps=30,
                 show_source=True,  # 是否显示原视频窗口（默认显示）
                 show_binary=True): # 是否显示二值化窗口（默认显示）
        # 输入源参数验证
        assert video_path is not None and os.path.exists(video_path), f"视频文件不存在: {video_path}"
        self.video_path = video_path
        
        # 帧率控制参数
        self.target_fps = target_fps
        self.delay = int(1000 / self.target_fps)
        
        # 初始化其他参数
        self.threshold_value = 60
        
        # 显示控制变量
        self.show_source = show_source
        self.show_binary = show_binary
        
        # 播放模式控制
        self.play_mode = 'continuous'  # 'continuous'（连续）/ 'step'（步进）
        self.current_frame = 0         # 当前帧号
        self.total_frames = 0          # 视频总帧数
        
        # 模式控制变量（二维码/线/漂移模式）
        self.mode = 'line'  # 默认线模式
        self.qr_detector = cv2.QRCodeDetector()  # 二维码检测器
        self.wurenji_center = (320, 320)  # 无人机中心点（图块4-5合并中心，默认640x480分辨率）
        
        # 启动视频处理流程
        self.start_video()

    def _open_source(self):
        cap = cv2.VideoCapture(self.video_path)
        print(f"=== 打开视频文件成功: {self.video_path} ===")
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频总帧数: {self.total_frames}")
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {self.video_path}")
        return cap

    def _get_source_params(self, cap):
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"输入源参数: {actual_width}x{actual_height}, 输入FPS: {input_fps:.1f}")
        print(f"目标显示帧率: {self.target_fps} FPS（每帧等待{self.delay}毫秒）")
        block_height = actual_height // 6
        block4_center_y = 3 * block_height + block_height // 2
        block5_center_y = 4 * block_height + block_height // 2
        self.wurenji_center = (actual_width // 2, (block4_center_y + block5_center_y) // 2)
        print(f"无人机中心点: {self.wurenji_center}")
        return actual_width, actual_height, input_fps

    def _print_usage(self):
        print("\n使用方法:")
        print("- 按 'q' 键: 停止并退出")
        print("- 按 's' 键: 切换播放模式（当前: {}）".format(self.play_mode))
        print("- 按 'a' 键: 上一帧（步进模式下）")
        print("- 按 'd' 键: 下一帧（步进模式下）")
        print("- 按 'g' 键: 跳转到任意帧")
        print(f"- 按 '+' 键: 增加二值化阈值（当前: {self.threshold_value}）")
        print(f"- 按 '-' 键: 减少二值化阈值（当前: {self.threshold_value}）")

    def _process_binary(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
        return binary

    def _draw_source_window(self, src):
        cv2.rectangle(src, (0, 0), (640, 70), (40, 40, 40), -1)
        cv2.putText(src, "NOT RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(src, f"Frame: {self.current_frame}/{self.total_frames}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Source (Original)", src)

    def _draw_binary_window(self, binary):
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        block_height = binary_bgr.shape[0] // 6
        for i in range(1, 6):
            cv2.line(binary_bgr, (0, i * block_height), (binary_bgr.shape[1], i * block_height), (0, 200, 200), 1)
        cv2.putText(
            binary_bgr, 
            f"Mode: {self.mode}", 
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0),
            1, 
            cv2.LINE_AA
        )
        center_x = binary_bgr.shape[1] // 2
        block_height = binary_bgr.shape[0] // 6
        selected_positions = [None] * 6
        block4_point = None
        block5_point = None
        for block_idx in range(6):
            y_start, y_end = block_idx * block_height, (block_idx + 1) * block_height
            roi_binary = binary[y_start:y_end, :]
            roi_inv = cv2.bitwise_not(roi_binary)
            contours, _ = cv2.findContours(roi_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = []
            valid_centers = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 1800:
                    continue
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    continue
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cnt[:, :, 1] += y_start
                cx_real = cx
                cy_real = cy + y_start
                distance_to_center = abs(cx_real - center_x)
                valid_contours.append((cnt, distance_to_center))
                valid_centers.append((cx_real, cy_real, distance_to_center))
                if block_idx == 3:
                    block4_point = (cx_real, cy_real)
                elif block_idx == 4:
                    block5_point = (cx_real, cy_real)
            if valid_contours:
                valid_contours.sort(key=lambda x: x[1])
                selected_cnt, min_distance = valid_contours[0]
                selected_positions[block_idx] = selected_cnt[0][0][0]
                cv2.drawContours(binary_bgr, [selected_cnt], -1, (255, 125, 255), 2)
                min_distance_point = None
                for cx_real, cy_real, dist in valid_centers:
                    if abs(dist - min_distance) < 1e-5:
                        min_distance_point = (cx_real, cy_real)
                        break
                if min_distance_point:
                    cx_real, cy_real = min_distance_point
                    cv2.circle(binary_bgr, (cx_real, cy_real), 3, (0, 0, 255), -1)
                    raw_offset = cx_real - center_x
                    color_raw = (0, 0, 255) if raw_offset > 0 else (255, 0, 0) if raw_offset < 0 else (0, 0, 0)
                    offset_text = f"O: {raw_offset:.1f}"
                    cv2.putText(binary_bgr, offset_text, (cx_real + 10, cy_real + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_raw, 1, cv2.LINE_AA)
        angles = []
        offsets = []
        for block_idx in range(5):
            if selected_positions[block_idx] is not None and selected_positions[block_idx + 1] is not None:
                y_start = block_idx * block_height
                y_end = (block_idx + 1) * block_height
                offset_x = selected_positions[block_idx] - selected_positions[block_idx + 1]
                turn_angle = np.arctan2(offset_x, block_height) * 180 / np.pi
                angles.append(turn_angle)
                offsets.append(offset_x)
                text_position = (10, y_end)
                color_angle = (0, 0, 255) if turn_angle > 0 else (255, 0, 0) if turn_angle < 0 else (0, 255, 0)
                angle_text = f"angle: {turn_angle:.2f}"
                cv2.putText(binary_bgr, angle_text, (text_position[0], text_position[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_angle, 1, cv2.LINE_AA)
        if block4_point is not None and block5_point is not None:
            mid_point_x = (block4_point[0] + block5_point[0]) // 2
            mid_point_y = (block4_point[1] + block5_point[1]) // 2
            cv2.circle(binary_bgr, (mid_point_x, mid_point_y), 5, (255, 0, 0), -1)
            cv2.line(binary_bgr, block4_point, block5_point, (0, 255, 0), 2)
            block4_center_y = 3 * block_height + block_height // 2
            block5_center_y = 4 * block_height + block_height // 2
            combined_center_x = center_x
            combined_center_y = (block4_center_y + block5_center_y) // 2
            self.wurenji_center = (combined_center_x, combined_center_y)
            cv2.circle(binary_bgr, self.wurenji_center, 5, (0, 255, 255), -1)
        cv2.imshow("Binary (Processed)", binary_bgr)
        return angles , offsets

    def _handle_keyboard(self, key):
        if key == ord('q'):
            return True
        if key == ord('s'):
            self.play_mode = 'step' if self.play_mode == 'continuous' else 'continuous'
            print(f"\n=== 切换播放模式至: {self.play_mode} ===")
            self._print_usage()
        if self.play_mode == 'step':
            if key == ord('a'):
                self.current_frame = max(0, self.current_frame - 1)
                print(f"=== 当前帧: {self.current_frame} ===")
            elif key == ord('d'):
                self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                print(f"=== 当前帧: {self.current_frame} ===")
        if key == ord('g'):
            try:
                print("=== 0-225为直线 ===")
                print("=== 226-540为第一弯道 ===")
                print("=== 541-730为第二弯道 ===")
                print("=== 731-977为第三弯道 ===")
                print("=== 978-1027为第四弯道 ===")
                target_frame = int(input("\n请输入要跳转的帧号（0到{}）: ".format(self.total_frames - 1)))
            except ValueError:
                print("=== 无效的帧号，请输入整数 ===")
                return False
            if 0 <= target_frame < self.total_frames:
                self.current_frame = target_frame
                self.play_mode = 'step'
                print(f"=== 跳转到帧: {self.current_frame} ===")
            else:
                print(f"=== 帧号超出范围（0到{self.total_frames - 1}） ===")
        if key in [ord('+'), ord('=')]:
            self.threshold_value = min(self.threshold_value + 5, 255)
            print(f"阈值增加至: {self.threshold_value}")
        elif key in [ord('-'), ord('_')]:
            self.threshold_value = max(self.threshold_value - 5, 0)
            print(f"阈值减少至: {self.threshold_value}")
        return False

    def qr_mode_process(self, binary):
        data, bbox, _ = self.qr_detector.detectAndDecode(binary)
        if data and bbox is not None:
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            bbox = np.int32(bbox)
            cv2.polylines(binary_bgr, [bbox], isClosed=True, color=(0, 255, 0), thickness=2)
            qr_center_x = int((bbox[0][0][0] + bbox[0][2][0]) / 2)
            qr_center_y = int((bbox[0][0][1] + bbox[0][2][1]) / 2)
            cv2.circle(binary_bgr, (qr_center_x, qr_center_y), 5, (0, 0, 255), -1)
            cv2.circle(binary_bgr, self.wurenji_center, 5, (0, 0, 255), -1)
            offset_x = qr_center_x - self.wurenji_center[0]
            offset_y = qr_center_y - self.wurenji_center[1]
            cv2.putText(
                binary_bgr, 
                f"QR Offset: ({offset_x}, {offset_y})", 
                (qr_center_x + 10, qr_center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1, 
                cv2.LINE_AA
            )
            cv2.imshow("Binary (Processed)", binary_bgr)

    def start_video(self):
        cap = self._open_source()
        actual_width, actual_height, input_fps = self._get_source_params(cap)
        self._print_usage()
        while cap.isOpened():
            delay = self.delay if self.play_mode == 'continuous' else 0
            if self.play_mode == 'continuous':
                ret, frame = cap.read()
                if ret:
                    self.current_frame += 1
                else:
                    print("=== 输入源已耗尽 ===")
                    break
            else:
                self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = cap.read()
                if not ret:
                    print(f"=== 无法读取帧: {self.current_frame} ===")
                    break
            src = frame.copy()
            binary = self._process_binary(frame)
            if self.show_source:
                self._draw_source_window(src)
            if self.show_binary and binary is not None:
                data, bbox, _ = self.qr_detector.detectAndDecode(frame)
                if data:
                    self.mode = 'qr'
                    self.qr_mode_process(binary)
                else:
                    angles , offsets = self._draw_binary_window(binary)
            key = cv2.waitKey(delay) & 0xFF
            if self._handle_keyboard(key):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    video_op = VideoDebugOp(
        video_path='E:/wurenji_workspace/test_workspace/original_30_20250627_204816.avi',
        target_fps=120,
        show_source=False,
        show_binary=True
    )