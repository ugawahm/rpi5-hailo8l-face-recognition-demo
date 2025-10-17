# rpi5-hailo8l-face-recognition-demo
树莓派5+hailo8l  人脸识别demo
树莓派系统镜像：raspberry pi os bookworm
hailort版本：4.20
本demo基于DeGirum PySDK, DeGirum Tools，项目地址：https://github.com/DeGirum/hailo_examples/tree/main
本项目代码经Gemini 2.5 pro优化
使用方法
1.项目根目录下创建/faces和/models目录
2.下载degirum hub（https://hub.degirum.com/public-models/degirum/hailo）提供的hef文件或将自己训练的hef文件导入models目录
3.激活degirum虚拟环境
4.运行main.py
运行示例：
python main.py                          # 使用 CSI 摄像头 (默认)
python main.py --input usb              # 使用 USB 摄像头
python main.py --input video.mp4        # 处理视频文件
python main.py --input photo.jpg        # 处理图片
python main.py --threshold 0.6          # 设置识别阈值为 0.6
python main.py --no-adaptive            # 禁用自适应阈值
