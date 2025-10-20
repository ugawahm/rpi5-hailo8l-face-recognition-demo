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
    def __init__(self, faces_dir="faces", models_dir="./models", detect_every_n_frames=3, 
                 cache_size=10, input_source="rpi", confidence_threshold=0.6, 
                 adaptive_threshold=True):
        """初始化人脸识别系统
        
        Args:
            input_source: 输入源 ('rpi', 'usb', 或文件路径)
            confidence_threshold: 人脸识别置信度阈值
            adaptive_threshold: 是否启用自适应阈值
        """
        init_start_time = time.time()
        print("🚀 初始化人脸识别系统...")
        
        # 初始化 DeGirum 本地模型Zoo
        print(f"📦 连接本地模型Zoo: {models_dir}")
        zoo_connect_start = time.time()
        zoo = dg.connect(dg.LOCAL, models_dir)
        print(f"✅ 连接本地模型Zoo成功，耗时: {time.time() - zoo_connect_start:.2f}s")
        
        # 加载人脸检测模型
        print("🔍 加载人脸检测模型...")
        detector_load_start = time.time()
        self.face_detector = zoo.load_model("scrfd_10g--640x640_quant_hailort_hailo8l_1")
        self.face_detector.input_letterbox_fill_color = (114, 114, 114)
        print(f"✅ 人脸检测模型加载成功，耗时: {time.time() - detector_load_start:.2f}s")
        
        # 加载人脸特征提取模型
        print("🧬 加载人脸特征提取模型...")
        encoder_load_start = time.time()
        self.face_encoder = zoo.load_model("arcface_r50")
        print(f"✅ 人脸特征提取模型加载成功，耗时: {time.time() - encoder_load_start:.2f}s")
        
        # 初始化输入源
        print("🎥 初始化输入源...")
        input_handler_init_start = time.time()
        self.input_handler = InputHandler(input_source)
        self.input_source_type = input_source
        print(f"✅ 输入源初始化成功，耗时: {time.time() - input_handler_init_start:.2f}s")
        
        # 人脸库管理
        print("📚 初始化人脸数据库管理器...")
        db_manager_init_start = time.time()
        self.face_database_manager = FaceDatabase(faces_dir, self.face_detector, self.face_encoder)
        self.face_database_manager.load_face_database() # 计时已在FaceDatabase内部
        print(f"✅ 人脸数据库管理器初始化成功，耗时: {time.time() - db_manager_init_start:.2f}s")
        
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

        # 首次调用模型标记
        # 在热身后，将这些设置为 False，这样主循环中的第一次真实检测就不会再打印“首帧耗时”
        self.first_detection_run = False 
        self.first_recognition_run = False 
        
        # 启动后台线程
        self._start_threads()

        # --- 新增：模型热身 (Model Warm-up) ---
        print("🔥 对Hailo-8L模型进行热身...")
        warmup_start = time.time()
        
        # 获取摄像头的默认分辨率，用于创建假的空白帧
        # 假设InputHandler已经初始化并可以获取分辨率
        # 你可能需要根据你的InputHandler实现来获取正确的尺寸
        # 如果是视频文件，可以尝试从视频读取第一帧来获取尺寸
        dummy_width = 640  # 默认值，请根据你的实际情况调整
        dummy_height = 480 # 默认值，请根据你的实际情况调整
        
        try:
            if self.input_source_type == "rpi":
                # Picamera2 启动后分辨率会稳定
                # 可以在 InputHandler 中添加方法来获取当前分辨率
                # 或者直接使用一个已知的分辨率，比如 640x480
                pass # 保持默认 dummy_width/height 或从 input_handler 获取
            elif self.input_source_type == "usb":
                 # USB摄像头可能默认分辨率
                 pass # 保持默认 dummy_width/height 或从 input_handler 获取
            else: # 视频文件
                temp_cap = cv2.VideoCapture(self.input_source_type)
                if temp_cap.isOpened():
                    dummy_width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    dummy_height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    temp_cap.release()
                else:
                    print(f"   ⚠️ 无法打开视频文件 {self.input_source_type} 获取分辨率，使用默认 {dummy_width}x{dummy_height}")
        except Exception as e:
            print(f"   ⚠️ 获取输入源分辨率失败: {e}，使用默认 {dummy_width}x{dummy_height}")

        dummy_frame = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8) 
        
        # 1. 热身人脸检测模型
        try:
            print("   -> 热身人脸检测模型...")
            _ = self.face_detector(dummy_frame)
            print("   ✅ 人脸检测模型热身完成。")
        except Exception as e:
            print(f"   ❌ 人脸检测模型热身失败: {e}")
            
        # 2. 热身人脸特征提取模型 (需要裁剪出“假”的人脸区域)
        # 确保裁剪区域不为空，且足够大供模型处理
        dummy_face_crop_h = min(100, dummy_height)
        dummy_face_crop_w = min(100, dummy_width)
        dummy_face_crop = dummy_frame[0:dummy_face_crop_h, 0:dummy_face_crop_w] 

        if dummy_face_crop.size > 0 and dummy_face_crop_h > 0 and dummy_face_crop_w > 0:
            try:
                print("   -> 热身人脸特征提取模型...")
                _ = self.face_encoder(dummy_face_crop)
                print("   ✅ 人脸特征提取模型热身完成。")
            except Exception as e:
                print(f"   ❌ 人脸特征提取模型热身失败: {e}")
        else:
            print("   ⚠️ 无法创建有效假人脸区域，跳过特征提取模型热身。")

        print(f"🔥 模型热身总耗时: {time.time() - warmup_start:.2f}s")
        # --- 热身结束 ---

        print(f"🚀 人脸识别系统初始化总耗时: {time.time() - init_start_time:.2f}s")


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
                detection_start = time.time()
                result = self.face_detector(frame)
                
                # if self.first_detection_run:
                #     print(f"⏱️ 首帧人脸检测耗时: {time.time() - detection_start:.3f}s (热身后)")
                #     self.first_detection_run = False
                
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
                    encode_start = time.time()
                    feature_vector = self.extract_features_with_cache(face_crop, bbox)
                    # if self.first_recognition_run:
                    #      print(f"⏱️ 首帧人脸特征提取耗时: {time.time() - encode_start:.3f}s (热身后)")
                    #      self.first_recognition_run = False
                    
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
        
        # 检查是否为静态图片源
        is_static_image = self.input_handler.is_static_image_source()
        processed_static_image = False # 标记静态图片是否已处理并显示过一次
        
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
                # 对于静态图片，只在第一次循环时发送一次
                if (self.frame_counter % self.detect_every_n_frames == 0 and not is_static_image) or \
                   (is_static_image and not processed_static_image):
                    if not self.detection_queue.full():
                        self.detection_queue.put((frame.copy(), frame_id))
                
                # 从结果队列获取识别结果
                current_detections = []
                captured_faces_for_save = []
                
                # 尝试从队列获取最新结果，如果队列为空则使用上次的结果
                # 对于静态图片，需要等待识别线程完成
                if is_static_image and not processed_static_image:
                    # 对于静态图片，等待识别结果，直到队列中有数据
                    try:
                        results, result_frame_id = self.result_queue.get(timeout=5) # 增加超时
                        # print(f"Static image: Got results for frame_id {result_frame_id}")
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
                        processed_static_image = True # 标记已处理
                    except Exception as e:
                        print(f"⚠️ 静态图片识别超时或错误: {e}. 显示原始图片。")
                        current_detections = [] # 如果超时，不显示任何检测框
                elif not self.result_queue.empty():
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
                if is_static_image and processed_static_image:
                    key = cv2.waitKey(0) & 0xFF 
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
                    continue
                elif is_static_image and not processed_static_image:
                    delay = 1 
                else:
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
