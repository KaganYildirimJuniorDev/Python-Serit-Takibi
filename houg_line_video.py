import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
import time
import Houg_Line_SollamaProtokolu 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


model = YOLO("yolov8n.pt") 

def combine_lane_centers(oneCenter=None, twoCenter=None, thirdCenter=None):
    """
    Üç farklı merkez tahminini ağırlıklı ortalamayla birleştirir.
    Eğer birden fazla merkez yoksa mevcut olanı döner.

    Ağırlıklar:
    - Sliding Window: 0.3
    - Canny + Hough: 0.5
    - Good Features to Track: 0.2
    """

    oneCenter = float(oneCenter)
    twoCenter = float(twoCenter)
    thirdCenter = float(thirdCenter)


    combined_center = (((oneCenter * 6) +(twoCenter * 1) + (thirdCenter * 3))/12)+180 # Bu Kısma Ekleme Yapman Gerek, Yanlış Çalışıyor. Aslında Fonksiyonu Güncelledim Ancak Yerleştirmeyi Beceremedim
    return int(combined_center)




# Kenar tespiti yapan fonksiyon
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 30, 60)
    return canny

# Şeritleri belirlemek için ilgi alanı maskesi
def region_of_interest(image):
    height = image.shape[0]
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# Çizgileri ekrana çizen fonksiyon
def display_lanes(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)  
    return line_image

# Çizgi koordinatlarını belirleyen fonksiyon
def make_coordinates(image, line_parameters):
    if line_parameters is None:
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 3 / 5)
    if abs(slope) < 1e-3:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2], dtype=int)

# Şeritleri belirlemek için ortalama hesaplayan fonksiyon
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope, intercept = parameters
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_lane = None
    right_lane = None
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_lane = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_lane = make_coordinates(image, right_fit_average)
    lanes = []
    if left_lane is not None:
        lanes.append(left_lane)
    if right_lane is not None:
        lanes.append(right_lane)
    return np.array(lanes)

# İki şeridin ortasını bulan fonksiyon
def find_lane_center(lanes):
    if len(lanes) != 2:
        return None  
    left_lane, right_lane = lanes
    x1_left, y1_left, x2_left, y2_left = left_lane
    x1_right, y1_right, x2_right, y2_right = right_lane
    mid_x = (x1_left + x1_right) // 2
    mid_y = (y1_left + y1_right) // 2
    return (mid_x, mid_y) # İlk Kontrol Mekanizması



# İlk Kontrol Mekanizması Şuan Sona Erdi. Bu kontrol mekanizması tek başına yeterli görülebilir, ancak ufak kusurlar olduğunda sistemin
# devre dışı kalmasını istemedğimizden ikinci ve üçüncü kontol mekanizmasına bakmamız gerekli. Sıra ikinci kontrol mekanizmasında



def make_circle(mask, frame2):
    corners = cv2.goodFeaturesToTrack(mask, 500, 0.01, 0.01)
    laneCenterMethod2 = None

    if corners is not None:
        corners = np.intp(corners) 
        points_left = []
        points_right = []

        # Dairelerin koordinatlarını iki grup halinde ayır
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame2, (int(x), int(y)), 7, (100, 250, 250), -1)

            # Y eksenine göre solda mı sağda mı olduğuna karar ver
            if x < frame2.shape[1] / 2:
                points_left.append((x, y))  # Sol grup
            else:
                points_right.append((x, y))  # Sağ grup

        # Soldaki ve sağdaki noktaların ortalama koordinatlarını bul
        if points_left and points_right:
            left_center = np.mean(points_left, axis=0)
            right_center = np.mean(points_right, axis=0)

            # Her iki grubun orta noktalarının ortasını al
            midpoint = ((left_center[0] + right_center[0]) / 2, (left_center[1] + right_center[1]) / 2)

            # Genel orta noktayı işaretle
            cv2.circle(frame2, (int(midpoint[0]), int(700)), 7, (0, 255, 0), -1) # İkinci Kontrol Mekanizması
            laneCenterMethod2 = midpoint[0]
    
    return frame2, laneCenterMethod2


# İkinci Kontrol Mekanizmasıda sona erdi. İkinci kontrol mekanizması tek başına yeterli olmayan bir sistem
# ancak başka bir kontrol mekanizmasının doğru karar vermesini sağlar. Sıra üçüncü kontrol mekanizmasında.


def perspective_lane(frame3):
    width, height = 800, 400

    pts1 = np.float32([[270, 430], [1000, 430], [270, 700], [1000, 700]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    Frameoutput = cv2.warpPerspective(frame3, matrix, (width, height))

    # Yeni perspektif görüntüdeki örnek bir nokta


    return Frameoutput

def perspective_lane_before(mainx):
    width, height = 800, 400

    pts1 = np.float32([[270, 430], [1000, 430], [270, 700], [1000, 700]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    mainx = None
    mainy = None
    new_point = np.array([[[mainx, 350]]], dtype='float32')

    # Ters matris hesapla
    inverse_matrix = np.linalg.inv(matrix)

    # Orijinal görüntüdeki karşılığı
    original_point = cv2.perspectiveTransform(new_point, inverse_matrix)
    original_x, original_y = original_point[0][0]

    print(f"Yeni görüntüde (mainx,350) noktası orijinal görüntüde: ({original_x:.2f}, {original_y:.2f})")

    return original_x


def nothnig(x):
    pass

cv2.namedWindow("Trackbar")
cv2.createTrackbar("L-H","Trackbar",0,255,nothnig)
cv2.createTrackbar("L-S","Trackbar",0,255,nothnig)
cv2.createTrackbar("L-V","Trackbar",0,255,nothnig)
cv2.createTrackbar("U-H","Trackbar",0,255,nothnig)
cv2.createTrackbar("U-S","Trackbar",0,255,nothnig)
cv2.createTrackbar("U-V","Trackbar",0,255,nothnig)
cv2.setTrackbarPos("L-H","Trackbar",0)
cv2.setTrackbarPos("L-S","Trackbar",0)
cv2.setTrackbarPos("L-V","Trackbar",115)

cv2.setTrackbarPos("U-H","Trackbar",255)
cv2.setTrackbarPos("U-S","Trackbar",250)
cv2.setTrackbarPos("U-V","Trackbar",255)




# Önceki şeritleri saklamak için değişken
previous_lanes = None
carx = 630

# Video işleme
cap = cv2.VideoCapture("test2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame3 = np.copy(frame)
    l_h = cv2.getTrackbarPos("L-H","Trackbar")
    l_s = cv2.getTrackbarPos("L-S","Trackbar")
    l_v = cv2.getTrackbarPos("L-V","Trackbar")
    u_h = cv2.getTrackbarPos("U-H","Trackbar")
    u_s = cv2.getTrackbarPos("U-S","Trackbar")
    u_v = cv2.getTrackbarPos("U-V","Trackbar")
    
    lower_hsv = np.array([l_h,l_s,l_v])
    upper_hsv = np.array([u_h,u_s,u_v])
    perspective = perspective_lane(frame3)
    perspective_hsv = cv2.cvtColor(perspective,cv2.COLOR_BGR2HSV)
    perspective_hsv = cv2.inRange(perspective_hsv,lower_hsv,upper_hsv)
    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 2:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame3, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Araba", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        elif cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame3, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "insan", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    histogram = np.sum(perspective_hsv[perspective_hsv.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint


    y = 470
    lx = []
    rx = []
    msk = perspective_hsv.copy()

    while y > 0:
        # Sol pencere
        l_start = max(left_base - 50, 0)
        l_end = min(left_base + 50, perspective_hsv.shape[1])
        l_img = perspective_hsv[y-40:y, l_start:l_end]

        l_contours, _ = cv2.findContours(l_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if l_contours:
            c = max(l_contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                lx.append(l_start + cx)
                left_base = l_start + cx

        # Sağ pencere
        r_start = max(right_base - 50, 0)
        r_end = min(right_base + 50, perspective_hsv.shape[1])
        r_img = perspective_hsv[y-40:y, r_start:r_end]

        r_contours, _ = cv2.findContours(r_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if r_contours:
            c = max(r_contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                rx.append(r_start + cx)
                right_base = r_start + cx

        cv2.rectangle(msk, (left_base - 50, y), (left_base + 50, y - 40), (255, 255, 255), 2)
        cv2.rectangle(msk, (right_base - 50, y), (right_base + 50, y - 40), (255, 255, 255), 2)
        middle_x = int((left_base+right_base) / 2)
        cv2.rectangle(msk, (middle_x - 10, 350), (middle_x + 10, 370), (255, 255, 255), -1) #üçüncü Kontrol Mekanizması
        laneCenterMethod3 = middle_x # Üçüncü Kontrol Mekanizmasının orta noktası, bu fonksiyonda kullanılacak.
        y -= 40



    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    frame2 = np.copy(frame)
    frame2, laneCenterMethod2 = make_circle(cropped_image,frame2)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    lines2 = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)

    
    if lines is not None:
        averaged_lines = average_slope_intercept(frame, lines)
        if len(averaged_lines) == 2:
            previous_lanes = averaged_lines
    elif previous_lanes is not None:
        averaged_lines = previous_lanes

    if previous_lanes is not None:
        laneCenterMethod1 = find_lane_center(previous_lanes)
        line_image = display_lanes(frame, previous_lanes)
        frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        
        height, width, _ = frame.shape
        bar_y = height - 50  
        cv2.rectangle(frame, (320, bar_y - 10), (width - 320, bar_y + 10), (100, 50, 50), -1)


        yolokontrol = None
        yolokontrol2 = None
        cv2.rectangle(frame,(350,bar_y-200),(width-350,bar_y-30), (255, 255, 50), 2)  #Kontrol Alanı Oluşturuluyor. Bu Alan İçerisinde Herhangi Bir Engel Varsa, duruma göre (engel Hareketli mi) ya sollama programını başlatıyor,
        cv2.putText(frame,"Kontrol Alani", (520,(bar_y-160)),cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 50), 1, cv2.LINE_AA, False) #ya da bekleme programını başlatıyor
        if (yolokontrol == True):
            #Yolo Kontrol 1 İle alan içinde cisim tespiti yapılır.
            if (yolokontrol2 == True):
                #Yolo Kontrol 2 İle Alan İçindeki cismin durumu hakkında incelelme yapılır.
                pass
            else:
                time.sleep(yolokontrol == False) #Yolo Kontrol1 olana kadar bekle.
        else:
            #Alan İçinde Bir Cisim Yoksa Normal Protokol Devam Eder
            pass




        
        
        if laneCenterMethod1 is not None:
            combine_x = combine_lane_centers(laneCenterMethod1[0],laneCenterMethod2 ,laneCenterMethod3)
            cv2.circle(frame, (laneCenterMethod1[0], bar_y), 15, (0, 0, 255), -1)  
            cv2.circle(frame, (combine_x, bar_y), 30, (0, 255, 255), 0)
            if combine_x - carx > 50:
                # Araba Şuanda kontrollerin orta noktasına göre solda kalmakta, bu durumu düzeltmek için araba sağa yönelmeli
                # Gereli Kütüphaneler Kullanılarak Rediktörlü motora güç verecek. https://olymposdesign.com/makaleler/guc-hesaplamalari-motor-reduktor import RPi.GPIO as GPIO
                pass
            elif carx - combine_x < 50:
                # Araba Şuanda kontrollerin orta noktasına göre sağda kalmakta, bu durumu düzeltmek için araba sola yönelmeli
                # Gereli Kütüphaneler Kullanılarak Rediktörlü motora güç verecek. https://olymposdesign.com/makaleler/guc-hesaplamalari-motor-reduktor import RPi.GPIO as GPIO
                pass
            else:
                #Araba şuan ortalama hareket etmekte, bir güç uygulanmasına gerek yok.
                pass




    cv2.imshow("1.Kontrol", frame)
    cv2.imshow("2.Kontrol", frame2)
    cv2.imshow("3.Kontrol", msk)


    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# Kod Bu Haliyle Kullanıma Hazır. Modifiye Edilebilir (Okunulabilirlik)