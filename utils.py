import cv2

def bbox_iou(box1, box2):
    """优化的 IoU 计算"""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def get_quality_indicator_color(quality):
    """根据质量指标返回颜色"""
    brightness = quality['brightness']
    blur = quality['blur']
    
    if 100 <= brightness <= 150 and blur > 100:
        return (0, 255, 0)  # 绿色：优秀
    elif 80 <= brightness <= 180 and blur > 50:
        return (255, 255, 0)  # 黄色：良好
    else:
        return (255, 165, 0)  # 橙色：较差

def draw_help_text(frame):
    """绘制帮助信息"""
    help_texts = [
        "Controls:",
        "S - Capture & Add Face",
        "ESC - Cancel Input",
        "Q - Quit",
    ]
    
    y_offset = frame.shape[0] - 120
    for i, text in enumerate(help_texts):
        cv2.putText(frame, text, (10, y_offset + i * 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
