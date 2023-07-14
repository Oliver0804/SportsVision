import cv2
import numpy as np
from collections import deque
from imutils.video import VideoStream
import argparse
import imutils
import time

# 建立參數解析器
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="影片檔案的路徑（可選）")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="追蹤點的最大數量")
args = vars(ap.parse_args())

# 定義"綠色"球的HSV顏色範圍，並初始化追蹤點的列表
greenLower = (90, 80, 150)
greenUpper = (120, 200, 255)
pts = deque(maxlen=args["buffer"])

# 如果沒有提供影片路徑，則使用攝影機
if not args.get("video", False):
    vs = VideoStream(src=0).start()
# 否則，使用提供的影片檔案
else:
    vs = cv2.VideoCapture(args["video"])

# 等待攝影機或影片讓其準備
time.sleep(2.0)

# 持續迴圈
while True:
    # 讀取當前影格
    frame = vs.read()

    # 從VideoCapture或VideoStream中取得影格
    frame = frame[1] if args.get("video", False) else frame

    # 如果是從影片中讀取且未取得影格，表示影片結束
    if frame is None:
        break

    # 調整影格大小、進行模糊處理並轉換為HSV色彩空間
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 建立遮罩以便檢測"綠色"區域，然後進行膨脹和腐蝕操作以去除小的斑點
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 尋找輪廓並初始化球的中心點
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # 只有當至少有一個輪廓時才執行以下操作
    if len(cnts) > 0:
        # 找到面積最大的輪廓，計算其最小外接圓和重心
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 只有當半徑大於一定大小時才執行以下操作
        if radius > 10:
            # 繪製球的外接圓和重心
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # 更新追蹤點列表
    pts.appendleft(center)

    # 迴圈遍歷追蹤點集合
    for i in range(1, len(pts)):
        # 如果其中一個追蹤點為None，則忽略
        if pts[i - 1] is None or pts[i] is None:
            continue

        # 計算線的粗細並繪製連接線
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # 顯示影格
    cv2.imshow("Frame", frame)

    # 按下 'q' 鍵結束迴圈
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 如果不是使用影片檔案，停止攝影機影片流
if not args.get("video", False):
    vs.stop()
# 否則，釋放攝影機
else:
    vs.release()

# 關閉所有視窗
cv2.destroyAllWindows()
