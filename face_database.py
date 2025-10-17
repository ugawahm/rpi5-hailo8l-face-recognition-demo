import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

class FaceDatabase:
    def __init__(self, faces_dir, face_detector, face_encoder):
        self.faces_dir = Path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.face_detector = face_detector
        self.face_encoder = face_encoder
        self.face_database = {}
        self.face_db_normalized = {}

    def load_face_database(self):
        """从文件夹加载人脸库"""
        print(f"📂 从 {self.faces_dir}/ 加载人脸库...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        face_images = [f for f in self.faces_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not face_images:
            print(f"⚠️  {self.faces_dir}/ 文件夹为空")
            print(f"💡 请添加人脸图片或使用 's' 键实时添加")
            return
        
        for img_path in face_images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                stem = img_path.stem
                is_cropped_face = '_face_' in stem
                
                if is_cropped_face:
                    face_crop = img_rgb
                else:
                    result = self.face_detector(img_rgb)
                    if not hasattr(result, 'results') or len(result.results) == 0:
                        print(f"⚠️  未检测到人脸: {img_path.name}")
                        continue
                    
                    bbox = result.results[0]['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    h, w = img_rgb.shape[:2]
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                    
                    face_crop = img_rgb[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                
                feature_vector = self.extract_features(face_crop)
                if feature_vector is None:
                    continue
                
                name = stem.split('_face_')[0] if '_face_' in stem else stem
                
                self.face_database[name] = feature_vector
                self.face_db_normalized[name] = feature_vector / np.linalg.norm(feature_vector)
                print(f"✅ 已加载: {name}")
                
            except Exception as e:
                print(f"❌ 处理失败 {img_path.name}: {e}")
        
        print(f"🎯 人脸库加载完成，共 {len(self.face_database)} 个身份")
    
    def extract_features(self, face_crop):
        """提取人脸特征向量"""
        try:
            if face_crop.shape[0] < 50 or face_crop.shape[1] < 50:
                return None
            
            feature_result = self.face_encoder(face_crop)
            
            if hasattr(feature_result, 'results'):
                results = feature_result.results
                if isinstance(results, list) and len(results) > 0:
                    first = results[0]
                    if isinstance(first, dict):
                        feature_vector = np.array(
                            first.get('data') or first.get('embedding') or first.get('features')
                        ).flatten()
                    else:
                        feature_vector = np.array(first).flatten()
                else:
                    feature_vector = np.array(results).flatten()
            else:
                feature_vector = np.array(feature_result).flatten()
            
            norm = np.linalg.norm(feature_vector)
            return feature_vector / norm if norm > 0 else None
            
        except Exception as e:
            # print(f"Error extracting features: {e}")
            return None

    def recognize_face(self, face_features, threshold):
        """识别人脸（向量化计算）"""
        if not self.face_db_normalized:
            return "Unknown", 0.0
        
        query_norm = face_features / np.linalg.norm(face_features)
        
        names = list(self.face_db_normalized.keys())
        db_features = np.array([self.face_db_normalized[n] for n in names])
        similarities = db_features @ query_norm
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity < threshold:
            return "Unknown", float(best_similarity)
        
        return names[best_idx], float(best_similarity)
    
    def save_face_to_database(self, face_crop, feature_vector, name):
        """保存人脸到数据库"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_face_{timestamp}.jpg"
            filepath = self.faces_dir / filename
            
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), face_bgr)
            
            self.face_database[name] = feature_vector
            self.face_db_normalized[name] = feature_vector / np.linalg.norm(feature_vector)
            
            print(f"✅ 已保存: {name} -> {filename}")
            return True
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
