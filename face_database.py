import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class FaceDatabase:
    def __init__(self, faces_dir, face_detector, face_encoder):
        self.faces_dir = Path(faces_dir)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.face_detector = face_detector
        self.face_encoder = face_encoder
        self.face_database = {}  # 存储原始特征向量
        self.face_db_normalized = {}  # 存储归一化特征向量，用于识别
        
        self.features_data_path = self.faces_dir / "face_features.npy"
        self.names_data_path = self.faces_dir / "face_names.json"
        
        # 用于缓存验证的元数据文件
        self.metadata_path = self.faces_dir / "face_metadata.json"

    def load_face_database(self):
        """
        从文件夹加载人脸库。
        优先从缓存文件加载。如果缓存文件不存在、损坏或与图片目录不一致，则从图片重新生成。
        """
        print(f"📂 从 {self.faces_dir}/ 加载人脸库...")
        
        # 1. 尝试从缓存加载
        if self._load_from_cache():
            print(f"✅ 从缓存成功加载人脸库，共 {len(self.face_database)} 个身份")
            return
        
        # 2. 如果缓存加载失败或验证不通过，则从图片重新生成
        print("⚠️ 缓存文件不存在、损坏或与图片目录不一致，将从图片文件重新生成。")
        self._rebuild_from_images()
        
        print(f"🎯 人脸库加载完成，共 {len(self.face_database)} 个身份")
        
        # 3. 如果从图片重新生成后有数据，则保存到缓存
        if self.face_database:
            self._save_to_cache()
        else:
            self._clear_cache_files() # 如果没有任何人脸，则清理缓存

    def _load_from_cache(self):
        """尝试从 .npy, .json 和 metadata 文件加载特征，并进行简单验证。"""
        if not (self.features_data_path.exists() and 
                self.names_data_path.exists() and 
                self.metadata_path.exists()):
            return False # 缓存文件不完整
        
        try:
            loaded_features = np.load(self.features_data_path)
            with open(self.names_data_path, 'r', encoding='utf-8') as f:
                loaded_names = json.load(f)
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # 缓存一致性初步检查
            if not (len(loaded_features) == len(loaded_names) and len(loaded_features) == metadata.get("num_faces", 0)):
                print("❌ 缓存文件内容不一致。")
                return False
            
            # 进一步验证：检查文件数量和最后修改时间 (简单粗暴但有效)
            current_image_files = [f for f in self.faces_dir.iterdir() 
                                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} and '_face_' in f.stem]
            
            if len(current_image_files) != metadata.get("num_image_files", 0):
                print("❌ 图片文件数量与缓存记录不符。")
                return False
            
            # 检查是否有更新的图片文件 (只检查数量)
            # 如果通过验证，则填充数据库
            self.face_database = {name: features for name, features in zip(loaded_names, loaded_features)}
            self.face_db_normalized = {name: features / np.linalg.norm(features) 
                                       for name, features in self.face_database.items()}
            return True
            
        except Exception as e:
            print(f"❌ 从缓存加载特征数据文件失败: {e}")
            self._clear_cache_files() # 损坏的缓存清理
            return False

    def _rebuild_from_images(self):
        """从图片文件重新生成人脸数据库。"""
        self.face_database = {}
        self.face_db_normalized = {}
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        face_images = [f for f in self.faces_dir.iterdir() 
                      if f.suffix.lower() in image_extensions and '_face_' in f.stem]
        
        if not face_images:
            print(f"⚠️  {self.faces_dir}/ 文件夹中没有人脸图片或标记不正确。")
            return
        
        temp_face_database = {} # 临时存储，用于构建数据库
        
        for img_path in face_images:
            try:
                name_part = img_path.stem.split('_face_')[0]
                
                print(f"🔄 处理图片并提取特征: {img_path.name}")
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"❌ 无法读取图片: {img_path.name}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                face_crop = img_rgb 
                
                feature_vector = self.extract_features(face_crop)
                if feature_vector is None:
                    print(f"❌ 无法从 {img_path.name} 提取特征，跳过。")
                    continue
                
                temp_face_database[name_part] = feature_vector
                
            except Exception as e:
                print(f"❌ 处理 {img_path.name} 失败: {e}")
        
        self.face_database = temp_face_database
        self.face_db_normalized = {name: features / np.linalg.norm(features) 
                                   for name, features in self.face_database.items()}
        
        # 更新元数据
        current_image_files = [f for f in self.faces_dir.iterdir() 
                                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} and '_face_' in f.stem]
        metadata = {
            "num_faces": len(self.face_database),
            "num_image_files": len(current_image_files),
            "last_rebuild_time": datetime.now().isoformat()
        }
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)


    def _save_to_cache(self):
        """将当前内存中的人脸数据库保存到缓存文件。"""
        try:
            if not self.face_database:
                self._clear_cache_files()
                return

            current_names = list(self.face_database.keys())
            current_features_array = np.array(list(self.face_database.values()))
            
            np.save(self.features_data_path, current_features_array)
            with open(self.names_data_path, 'w', encoding='utf-8') as f:
                json.dump(current_names, f, ensure_ascii=False, indent=4)
            
            # 更新元数据
            current_image_files = [f for f in self.faces_dir.iterdir() 
                                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} and '_face_' in f.stem]
            metadata = {
                "num_faces": len(self.face_database),
                "num_image_files": len(current_image_files),
                "last_update_time": datetime.now().isoformat()
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=4)

            print(f"✅ 人脸特征已同步保存到缓存文件。")
        except Exception as e:
            print(f"❌ 保存人脸特征数据文件失败: {e}")

    def _clear_cache_files(self):
        """清除所有缓存文件 (.npy, .json, metadata.json)"""
        if self.features_data_path.exists():
            self.features_data_path.unlink()
            print(f"🗑️ 已删除旧的特征文件: {self.features_data_path}")
        if self.names_data_path.exists():
            self.names_data_path.unlink()
            print(f"🗑️ 已删除旧的名称文件: {self.names_data_path}")
        if self.metadata_path.exists():
            self.metadata_path.unlink()
            print(f"🗑️ 已删除旧的元数据文件: {self.metadata_path}")


    def extract_features(self, face_crop):
        """提取人脸特征向量 (与原代码相同)"""
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
        """识别人脸（向量化计算）(与原代码相同)"""
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
        """保存人脸到数据库 (更新内存，然后保存到缓存)"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_face_{timestamp}.jpg"
            filepath = self.faces_dir / filename
            
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(filepath), face_bgr)
            
            print(f"✅ 已保存人脸图片: {name} -> {filename}")
            
            # 更新内存中的人脸库
            # 如果同一个name已经存在，则替换
            self.face_database[name] = feature_vector
            self.face_db_normalized[name] = feature_vector / np.linalg.norm(feature_vector)
            
            # 更新缓存文件
            self._save_to_cache()
            
            return True
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            return False
