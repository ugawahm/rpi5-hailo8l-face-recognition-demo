import cv2
import time
from pathlib import Path
from picamera2 import Picamera2

class InputHandler:
    def __init__(self, input_source):
        self.input_source_type = input_source
        self.cap = None
        self.camera = None
        self.static_image = None
        self._init_input_source(input_source)

    def _init_input_source(self, input_source):
        """初始化输入源"""
        if input_source == "rpi":
            print("📷 初始化 Picamera2 (CSI)...")
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"},
                controls={"FrameRate": 24}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1)
            print("✅ CSI 摄像头启动成功")
            
        elif input_source == "usb":
            print("📷 初始化 USB 摄像头...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 24)
            if not self.cap.isOpened():
                raise RuntimeError("❌ 无法打开 USB 摄像头")
            print("✅ USB 摄像头启动成功")
            
        else:
            input_path = Path(input_source)
            if not input_path.exists():
                raise FileNotFoundError(f"❌ 文件不存在: {input_source}")
            
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                print(f"🖼️ 加载图片: {input_source}")
                self.static_image = cv2.imread(str(input_path))
                if self.static_image is None:
                    raise RuntimeError(f"❌ 无法读取图片: {input_source}")
            else:
                print(f"🎥 加载视频: {input_source}")
                self.cap = cv2.VideoCapture(str(input_path))
                if not self.cap.isOpened():
                    raise RuntimeError(f"❌ 无法打开视频: {input_source}")
            print("✅ 文件加载成功")

    def capture_frame(self):
        """统一的帧捕获接口"""
        if self.camera is not None:
            return cv2.cvtColor(self.camera.capture_array(), cv2.COLOR_BGR2RGB)
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        elif self.static_image is not None:
            return cv2.cvtColor(self.static_image, cv2.COLOR_BGR2RGB)
        return None

    def is_static_image_source(self):
        """判断是否为静态图片源"""
        return self.static_image is not None

    def release(self):
        """释放资源"""
        if self.camera:
            self.camera.stop()
        if self.cap:
            self.cap.release()
