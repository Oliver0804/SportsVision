import cv2
import numpy as np

# 建立影片擷取物件
cap = cv2.VideoCapture('./videos/test.mp4')

while True:
    # 讀取影格
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 將影格轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 獲取影像的寬度和高度
    height, width = gray.shape[:2]

    # 計算畫面中間1/3的範圍
    start_x = width // 3
    end_x = width * 2 // 3

    # 只擷取畫面中間1/3的區域進行處理
    gray_roi = gray[:, start_x:end_x]

    # 套用模糊濾鏡以降低噪音
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # 偵測圓形，並調整閥值
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=30, param2=40, minRadius=20, maxRadius=50)

    # 確保至少偵測到一個圓形
    if circles is not None:
        # 將圓形參數的值轉換成整數
        circles = np.round(circles[0, :]).astype("int")

        # 繪製偵測到的圓形
        for (x, y, r) in circles:
            # 轉換圓心的座標到原始圖像的座標
            x += start_x
            # 繪製圓形
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)

    # 顯示影像
    cv2.imshow("Detected Circles", frame)
    
    # 按下 'q' 鍵結束迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放影片擷取物件與關閉視窗
cap.release()
cv2.destroyAllWindows()
