import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import keypointrcnn_resnet50_fpn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# COCO數據集中的類別名稱
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 我們關心的物件類別
TARGET_CLASSES = ['person', 'baseball', 'basketball', 'soccer ball', 'tennis ball', 'volleyball']

# 信心值閾值
DETECTION_THRESHOLD = 0.5

# 載入物件偵測模型q
model_detection = fasterrcnn_resnet50_fpn(pretrained=True)
model_detection = model_detection.eval()

# 載入人體骨架偵測模型
model_pose = keypointrcnn_resnet50_fpn(pretrained=True)
model_pose = model_pose.eval()

# 使用攝影機擷取影像
#vs = cv2.VideoCapture(0)
vs = cv2.VideoCapture('./videos/test.mp4')

# 獲取影片的幀率
fps = vs.get(cv2.CAP_PROP_FPS)

# 計算要跳過的幀數
start_time_in_seconds = 10
frames_to_skip = int(start_time_in_seconds * fps)

# 跳過指定的幀數
vs.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)


while True:
    # 讀取一幅影像
    ret, frame = vs.read()
    frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    if not ret:
        break

    # 轉換影像為 PyTorch tensor
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor_image = T.ToTensor()(rgb_image).unsqueeze(0)

    # 進行物件偵測
    with torch.no_grad():
        prediction = model_detection(tensor_image)

    # 複製一份原始影像來繪製結果
    output_image = frame.copy()

    for i, box in enumerate(prediction[0]['boxes']):
        # 檢查該物件的信心值是否大於設定的閾值
        if prediction[0]['scores'][i] > DETECTION_THRESHOLD:
            # 確認物件類別在我們關心的範圍內
            label = prediction[0]['labels'][i].item()
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            if label_name not in TARGET_CLASSES:
                continue

            # 取得物件的邊界框座標並繪製邊界框
            xmin, ymin, xmax, ymax = box.numpy().astype(int)
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # 繪製物件類別的名稱
            cv2.putText(output_image, label_name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 進行人體骨架偵測
    with torch.no_grad():
        pose_output = model_pose(tensor_image)

    keypoints = pose_output[0]['keypoints']

    for person_keypoints in keypoints:
        person_keypoints = person_keypoints.detach().numpy()

        for keypoint in person_keypoints:
            x, y, p = keypoint
            if p > DETECTION_THRESHOLD:
                cv2.circle(output_image, (int(x), int(y)), 3, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)

    # 顯示結果影像
    cv2.imshow("Output", output_image)
    key = cv2.waitKey(1) & 0xFF

    # 若按下 'q' 鍵則離開迴圈
    if key == ord('q'):
        break

# 釋放資源並關閉視窗
vs.release()
cv2.destroyAllWindows()
