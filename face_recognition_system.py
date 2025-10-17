import cv2
import numpy as np
import degirum as dg
import time
import threading
from collections import OrderedDict
from queue import Queue
from datetime import datetime
from pathlib import Path

from input_handler import InputHandler
from face_database import FaceDatabase
from quality_assessment import QualityAssessor
from utils import bbox_iou, get_quality_indicator_color, draw_help_text

class FaceRecognitionSystem:
    def __init__(self, faces_dir="./faces", models_dir="./models", detect_every_n_frames=5, 
                 cache_size=10, input_source="rpi", confidence_threshold=0.55, 
                 adaptive_threshold=True):
        """初始化程序
        
        Args:
            input_source: 输入源 ('rpi', 'usb', 或文件路径)
            confidence_threshold: 人脸识别置信度阈值
            adaptive_threshold: 是否启用自适应阈值
        """
        print("🚀 初始化...")
        
        # 初始化 DeGirum 本地模型Zoo
        print(f"📦 连接本地模型Zoo: {models_dir}")
        zoo = dg.connect(dg.LOCAL, models_dir)
        
        # 加载人脸检测模型
        print("🔍 加载人脸检测模型...")
        self.face_detector = zoo.load_model("scrfd_10g--640x640_quant_hailort_hailo8l_1")
        self.face_detector.input_letterbox_fill_color = (114, 114, 114)
        print("✅ 人脸检测模型加载成功")
        
        # 加载人脸特征提取模型
        print("🧬 加载人脸特征提取模型...")
        self.face_encoder = zoo.load_model("arcface_r50")
        print("✅ 人脸特征提取模型加载成功")
        
        # 初始化输入源
        self.input_handler = InputHandler(input_source)
        self.input_source_type = input_source
        
        # 人脸库管理
        self.face_database_manager = FaceDatabase(faces_dir, self.face_detector, self.face_encoder)
        self.face_database_manager.load_face_database()
        
        # 性能优化参数
        self.detect_every_n_frames = detect_every_n_frames
        self.frame_counter = 0
        self.last_detections = []
        self.last_captured_faces = []
        self.base_threshold = confidence_threshold
        self.adaptive_threshold_enabled = adaptive_threshold
        
        # 图像质量评估和自适应阈值
        self.quality_assessor = QualityAssessor(confidence_threshold, adaptive_threshold)
        
        # 特征缓存
        self.feature_cache = OrderedDict()
        self.cache_size = cache_size
        
        # 多线程队列
        self.detection_queue = Queue(maxsize=2)
        self.recognition_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        
        # 线程控制
        self.running = True
        self.detection_thread = None
        self.recognition_thread = None
        
        # FPS 计算
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # 输入模式控制
        self.input_mode = False
        
        # 启动后台线程
        self._start_threads()

    def _start_threads(self):
        """启动后台处理线程"""
        print("🔧 启动多线程处理...")
        
        # 检测线程
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.detection_thread.start()
        
        # 识别线程
        self.recognition_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self.recognition_thread.start()
        
        print("✅ 多线程启动成功")

    def _detection_worker(self):
        """后台人脸检测线程"""
        while self.running:
            try:
                # 从队列获取帧
                if self.detection_queue.empty():
                    time.sleep(0.001)
                    continue
                
                frame_data = self.detection_queue.get(timeout=0.1)
                if frame_data is None:
                    continue
                
                frame, frame_id = frame_data
                
                # 执行人脸检测
                result = self.face_detector(frame)
                detected_faces = []
                
                if hasattr(result, 'results') and len(result.results) > 0:
                    for detection in result.results:
                        try:
                            bbox = detection['bbox']
                            x1, y1, x2, y2 = map(int, bbox)
                            h, w = frame.shape[:2]
                            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                            
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size == 0:
                                continue
                            
                            # 评估图像质量
                            quality = self.quality_assessor.assess_image_quality(frame, (x1, y1, x2, y2))
                            
                            detected_faces.append({
                                'bbox': (x1, y1, x2, y2),
                                'crop': face_crop,
                                'quality': quality
                            })
                        except Exception as e:
                            # print(f"Error processing detection: {e}")
                            continue
                
                # 将检测结果发送到识别队列
                if not self.recognition_queue.full():
                    self.recognition_queue.put((detected_faces, frame_id))
                
            except Exception as e:
                # print(f"Detection worker error: {e}")
                continue

    def _recognition_worker(self):
        """后台人脸识别线程"""
        while self.running:
            try:
                if self.recognition_queue.empty():
                    time.sleep(0.001)
                    continue
                
                face_data = self.recognition_queue.get(timeout=0.1)
                if face_data is None:
                    continue
                
                detected_faces, frame_id = face_data
                recognition_results = []
                
                for face_info in detected_faces:
                    bbox = face_info['bbox']
                    face_crop = face_info['crop']
                    quality = face_info['quality']
                    
                    # 提取特征
                    feature_vector = self.extract_features_with_cache(face_crop, bbox)
                    if feature_vector is None:
                        continue
                    
                    # 计算自适应阈值
                    adaptive_threshold = self.quality_assessor.calculate_adaptive_threshold(quality)
                    
                    # 识别人脸
                    name, similarity = self.face_database_manager.recognize_face(feature_vector, adaptive_threshold)
                    confidence = similarity * 100
                    
                    recognition_results.append({
                        'bbox': bbox,
                        'name': name,
                        'confidence': confidence,
                        'quality': quality,
                        'threshold': adaptive_threshold,
                        'crop': face_crop,
                        'features': feature_vector
                    })
                
                # 将结果发送到结果队列
                if not self.result_queue.full():
                    self.result_queue.put((recognition_results, frame_id))
                
            except Exception as e:
                # print(f"Recognition worker error: {e}")
                continue

    def extract_features_with_cache(self, face_crop, bbox, iou_threshold=0.8):
        """使用缓存提取特征"""
        bbox_tuple = tuple(bbox)
        
        for cached_bbox, cached_features in list(self.feature_cache.items()):
            if bbox_iou(bbox_tuple, cached_bbox) > iou_threshold:
                self.feature_cache.move_to_end(cached_bbox)
                return cached_features

        features = self.face_database_manager.extract_features(face_crop)
        if features is not None:
            if len(self.feature_cache) >= self.cache_size:
                self.feature_cache.popitem(last=False)
            self.feature_cache[bbox_tuple] = features
        return features
    
    def update_fps(self):
        """更新FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

    def handle_face_capture_async(self, captured_faces):
        """异步处理人脸捕获输入"""
        if not captured_faces:
            print("❌ 未检测到人脸，请对准摄像头")
            self.input_mode = False
            return

        print(f"\n📸 检测到 {len(captured_faces)} 个人脸")

        if len(captured_faces) == 1:
            print("请输入姓名 (直接回车取消): ", end='', flush=True)
            try:
                name = input().strip()
            except (EOFError, KeyboardInterrupt):
                print("\n❌ 输入被中断")
                self.input_mode = False
                return

            if name:
                face_info = captured_faces[0]
                if self.face_database_manager.save_face_to_database(
                    face_info['crop'], 
                    face_info['features'], 
                    name
                ):
                    print(f"✅ 成功添加 {name} 到人脸库!")
            else:
                print("❌ 已取消")
        else:
            print("请选择要保存的人脸:")
            for i in range(len(captured_faces)):
                print(f"  [{i+1}] 人脸 {i+1}")
            print("  [0] 取消")

            try:
                choice_input = input("选择 (0-{}): ".format(len(captured_faces))).strip()
                if not choice_input:
                    print("❌ 已取消")
                    self.input_mode = False
                    return
                choice = int(choice_input)

                if 1 <= choice <= len(captured_faces):
                    print("请输入姓名 (直接回车取消): ", end='', flush=True)
                    name = input().strip()

                    if name:
                        face_info = captured_faces[choice - 1]
                        if self.face_database_manager.save_face_to_database(
                            face_info['crop'], 
                            face_info['features'], 
                            name
                        ):
                            print(f"✅ 成功添加 {name} 到人脸库!")
                    else:
                        print("❌ 已取消")
                else:
                    print("❌ 已取消")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("❌ 操作已取消")

        print()
        self.input_mode = False

    def run(self):
        """运行主循环"""
        print("\n🎬 开始实时人脸识别...")
        print("=" * 60)
        print("⌨️  按键说明:")
        print("   's'   - 捕获当前人脸并添加到数据库")
        print("   'ESC' - 取消输入 (在输入姓名时)")
        print("   'q'   - 退出程序")
        print("=" * 60)
        print(f"📹 输入源: {self.input_source_type}")
        print(f"⚙️  跳帧检测: 每 {self.detect_every_n_frames} 帧执行一次检测")
        print(f"🎯 基准阈值: {self.base_threshold:.2f}")
        print(f"🧠 自适应阈值: {'启用' if self.adaptive_threshold_enabled else '禁用'}")
        print(f"🔧 多线程处理: 启用")
        print()
        
        frame_id = 0
        
        try:
            while True:
                loop_start = time.time()
                frame = self.input_handler.capture_frame()
                
                if frame is None:
                    print("❌ 无法获取帧")
                    break
                
                self.frame_counter += 1
                frame_id += 1
                
                # 定期向检测队列发送帧
                if self.frame_counter % self.detect_every_n_frames == 0:
                    if not self.detection_queue.full():
                        self.detection_queue.put((frame.copy(), frame_id))
                
                # 从结果队列获取识别结果
                current_detections = []
                captured_faces_for_save = []
                
                if not self.result_queue.empty():
                    try:
                        results, result_frame_id = self.result_queue.get_nowait()
                        
                        for result in results:
                            bbox = result['bbox']
                            name = result['name']
                            confidence = result['confidence']
                            quality = result['quality']
                            
                            current_detections.append((bbox, name, confidence, quality))
                            
                            captured_faces_for_save.append({
                                'bbox': bbox,
                                'crop': result['crop'],
                                'features': result['features']
                            })
                        
                        self.last_detections = current_detections
                        self.last_captured_faces = captured_faces_for_save
                        
                    except:
                        current_detections = self.last_detections
                else:
                    current_detections = self.last_detections

                # 绘制检测框和信息
                for bbox, name, confidence, quality in current_detections:
                    x1, y1, x2, y2 = bbox
                    
                    # 根据识别结果选择颜色
                    if name != "Unknown":
                        color_rgb = (0, 255, 0)
                    else:
                        color_rgb = (255, 0, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_rgb, 2)
                    
                    # 显示识别结果
                    label = f"{name}: {confidence:.1f}%"
                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rgb, 2)
                    
                    # 显示质量指标（调试信息）
                    quality_color = get_quality_indicator_color(quality)
                    quality_text = f"Q: B{quality['brightness']:.0f} C{quality['contrast']:.0f} S{quality['blur']:.0f}"
                    cv2.putText(frame, quality_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)

                if self.input_mode:
                    mode_text = "Waiting for input... (Check terminal)"
                    cv2.putText(frame, mode_text, (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                # 显示状态信息
                self.update_fps()
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Database: {len(self.face_database_manager.face_database)} faces", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Detected: {len(current_detections)} face(s)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示自适应阈值信息
                if self.adaptive_threshold_enabled:
                    threshold_text = f"Threshold: {self.quality_assessor.current_threshold:.3f} (Base: {self.base_threshold:.2f})"
                    cv2.putText(frame, threshold_text, (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

                draw_help_text(frame)

                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("Face Recognition System - Multi-threaded", frame_bgr)

                # 动态调整延迟
                elapsed = time.time() - loop_start
                target_fps = 24
                delay = max(1, int((1.0 / target_fps - elapsed) * 1000))
                key = cv2.waitKey(delay) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s') and not self.input_mode:
                    captured_faces_snapshot = self.last_captured_faces.copy()
                    if not captured_faces_snapshot:
                        print("❌ 未检测到人脸，请对准摄像头")
                        continue

                    self.input_mode = True
                    thread = threading.Thread(
                        target=self.handle_face_capture_async,
                        args=(captured_faces_snapshot,)
                    )
                    thread.daemon = True
                    thread.start()

                # 静态图片模式
                if self.input_handler.is_static_image_source():
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\n⏹️  用户中断")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理资源...")
        self.running = False
        
        # 等待线程结束
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1)
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1)
        
        self.input_handler.release()
        cv2.destroyAllWindows()
        print("✅ 完成")
