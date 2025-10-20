import argparse
import traceback
from face_recognition_system import FaceRecognitionSystem

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='树莓派5 + Hailo-8L 人脸识别系统demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 使用 CSI 摄像头 (默认)
  python main.py --input usb              # 使用 USB 摄像头
  python main.py --input video.mp4        # 处理视频文件
  python main.py --input photo.jpg        # 处理图片
  python main.py --threshold 0.5          # 设置识别阈值为 0.5
  python main.py --no-adaptive            # 禁用自适应阈值
        """
    )
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='rpi',
        help='输入源: "rpi" (CSI摄像头), "usb" (USB摄像头), 或文件路径'
    )
    
    parser.add_argument(
        '--faces-dir',
        type=str,
        default='./faces',
        help='人脸数据库目录 (默认: ./faces)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./models',
        help='模型目录 (默认: ./models)'
    )
    
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=3,
        help='跳帧检测间隔 (默认: 3)'
    )
    
    parser.add_argument(
        '--cache-size',
        type=int,
        default=10,
        help='特征缓存大小 (默认: 10)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.55,
        help='人脸识别基准置信度阈值 (默认: 0.55)'
    )
    
    parser.add_argument(
        '--no-adaptive',
        action='store_true',
        help='禁用自适应阈值调整'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    print("=" * 60)
    print("🔷 树莓派5 + Hailo-8L 人脸识别系统demo")
    print("=" * 60)
    
    args = parse_args()
    
    try:
        system = FaceRecognitionSystem(
            faces_dir=args.faces_dir,
            models_dir=args.models_dir,
            detect_every_n_frames=args.skip_frames,
            cache_size=args.cache_size,
            input_source=args.input,
            confidence_threshold=args.threshold,
            adaptive_threshold=not args.no_adaptive
        )
        system.run()
    except Exception as e:
        print(f"❌ 错误: {e}")
        traceback.print_exc()
