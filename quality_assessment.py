import cv2
import numpy as np

class QualityAssessor:
    def __init__(self, base_threshold, adaptive_threshold_enabled, history_size=30):
        self.base_threshold = base_threshold
        self.adaptive_threshold_enabled = adaptive_threshold_enabled
        self.light_quality_history = []  # 光照质量历史
        self.blur_history = []           # 模糊度历史
        self.history_size = history_size # 历史记录大小
        self.current_threshold = base_threshold

    def assess_image_quality(self, frame, face_bbox=None):
        """评估图像质量（光照和清晰度）
        
        Returns:
            dict: {'brightness': float, 'contrast': float, 'blur': float}
        """
        # 如果提供了人脸区域，只评估该区域
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
        
        if roi.size == 0:
            return {'brightness': 128, 'contrast': 50, 'blur': 100}
        
        # 转换为灰度图
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if len(roi.shape) == 3 else roi
        
        # 亮度评估（均值）
        brightness = np.mean(gray)
        
        # 对比度评估（标准差）
        contrast = np.std(gray)
        
        # 清晰度评估（拉普拉斯方差，值越大越清晰）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        return {
            'brightness': float(brightness),
            'contrast': float(contrast),
            'blur': float(blur_score)
        }

    def calculate_adaptive_threshold(self, quality_metrics):
        """根据图像质量动态调整识别阈值
        
        Args:
            quality_metrics: 图像质量指标
            
        Returns:
            float: 调整后的阈值
        """
        if not self.adaptive_threshold_enabled:
            return self.base_threshold
        
        brightness = quality_metrics['brightness']
        contrast = quality_metrics['contrast']
        blur = quality_metrics['blur']
        
        # 更新历史记录
        self.light_quality_history.append(brightness)
        self.blur_history.append(blur)
        
        if len(self.light_quality_history) > self.history_size:
            self.light_quality_history.pop(0)
        if len(self.blur_history) > self.history_size:
            self.blur_history.pop(0)
        
        # 计算调整因子
        adjustment = 0.0
        
        # 1. 光照调整（理想亮度 100-150）
        if brightness < 80:  # 过暗
            adjustment -= 0.08
        elif brightness > 180:  # 过亮
            adjustment -= 0.05
        
        # 2. 对比度调整（理想对比度 > 40）
        if contrast < 30:  # 对比度低
            adjustment -= 0.06
        
        # 3. 清晰度调整（理想清晰度 > 100）
        if blur < 50:  # 模糊
            adjustment -= 0.10
        elif blur < 100:
            adjustment -= 0.05
        
        # 4. 稳定性调整（光照和清晰度波动）
        if len(self.light_quality_history) >= 10:
            light_std = np.std(self.light_quality_history[-10:])
            blur_std = np.std(self.blur_history[-10:])
            
            if light_std > 20:  # 光照不稳定
                adjustment -= 0.03
            if blur_std > 30:  # 清晰度不稳定
                adjustment -= 0.03
        
        # 计算新阈值（限制在合理范围内）
        new_threshold = np.clip(
            self.base_threshold + adjustment,
            self.base_threshold - 0.15,  # 最低不低于基准 0.15
            self.base_threshold + 0.05   # 最高不超过基准 0.05
        )
        
        self.current_threshold = new_threshold
        return new_threshold
