from l2cs import Pipeline, render
import cv2
import torch
from pathlib import Path

# 设置模型和摄像头参数
CWD = Path(__file__).parent  # 获取当前文件夹的路径，假设模型也在这个文件夹下
gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'fold14.pkl',
    arch='ResNet50',
    device=torch.device('cpu')  # 可以根据您的配置选择使用'cpu'或'gpu'
)

cap = cv2.VideoCapture(0)  # 默认0为电脑的内置摄像头

try:
    while True:
        ret, frame = cap.read()  # 读取一帧视频
        if not ret:
            break  # 如果无法获取帧，则退出循环

        # 使用模型处理获取的帧
        results = gaze_pipeline.step(frame)

        # 将处理结果渲染到帧上
        frame = render(frame, results)

        # 显示处理后的帧
        cv2.imshow('Gaze Tracking and Head Pose', frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()  # 释放摄像头资源
    cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

# python demo.py  --snapshot models/L2CSNet_gaze360.pkl --cam 0 
# python demo.py  --snapshot models/L2CSNet_gaze360.pkl --device gpu:0 --cam 0 