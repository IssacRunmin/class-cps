def calculate_iou_loss(ground_truth_box, bounding_box):
    # 提取图框的参数
    x1, y1, w1, h1 = ground_truth_box
    x2, y2, w2, h2 = bounding_box

    # 计算矩形框的四个边界坐标
    left1, right1, top1, bottom1 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
    left2, right2, top2, bottom2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2

    # 计算交集的坐标
    intersection_left = max(left1, left2)
    intersection_right = min(right1, right2)
    intersection_top = max(top1, top2)
    intersection_bottom = min(bottom1, bottom2)

    # 计算交集区域的面积
    intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)

    # 计算并集面积
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area

    # 计算IoU
    iou = intersection_area / union_area

    # 计算IoU损失
    iou_loss = 1 - iou

    return iou_loss


# 测试用例
ground_truth_box = [0, 0, 2, 2]  # 中心坐标 (0, 0), 长度 2, 宽度 2
bounding_box = [0, 1, 2, 2]  # 中心坐标 (1, 1), 长度 2, 宽度 2

iou_loss = calculate_iou_loss(ground_truth_box, bounding_box)
print("IoU损失:", iou_loss)




