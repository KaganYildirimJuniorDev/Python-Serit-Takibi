import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü alıp kenarlarını tespit eden fonksiyon

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Görüntüyü gri tona çeviriyoruz.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Gürültüyü azaltmak için Gauss bulanıklığı uyguluyoruz.
    canny = cv2.Canny(blur, 50, 150)  # Canny algoritması ile kenar tespiti yapıyoruz.
    return canny

# Yol çizgilerini tespit etmek için ilgi alanını belirleyen fonksiyon

def region_of_interest(image):
    height = image.shape[0]  # Görüntünün yüksekliğini alıyoruz.
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])  # Üçgen şeklinde ilgi alanı belirliyoruz.
    mask = np.zeros_like(image)  # Aynı boyutta tamamen siyah bir maske oluşturuyoruz.
    cv2.fillPoly(mask, polygon, 255)  # Maskeye üçgen bölgeyi beyaz olarak çiziyoruz.
    masked_image = cv2.bitwise_and(image, mask)  # Maskeyi görüntüye uygulayarak sadece ilgi alanını bırakıyoruz.
    return masked_image

# Çizgileri görüntüye ekleyen fonksiyon

def display_lanes(image, lines):
    line_image = np.zeros_like(image)  # Boş bir görüntü oluşturuyoruz.
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)  # Satırları 4 noktaya ayırıyoruz.
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Çizgileri yeşil renkte çiziyoruz.
    return line_image

# Çizgi koordinatlarını belirleyen fonksiyon

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters  # Doğrunun eğimi ve kesişim noktası alınıyor.
    y1 = image.shape[0]  # Çizginin alt noktası görüntünün en altı olacak.
    y2 = int(y1 * 3 / 5)  # Çizginin üst noktası görüntünün belirli bir kısmına kadar uzanacak.
    x1 = int((y1 - intercept) / slope)  # x koordinatlarını hesaplıyoruz.
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# Çizgi parametrelerini ortalama alarak yumuşatmak için kullanılan fonksiyon

def average_slope_intercept(image, lines):
    left_fit = []  # Sol şerit çizgileri için liste
    right_fit = []  # Sağ şerit çizgileri için liste
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Doğrunun eğimini ve kesişimini hesaplıyoruz.
        slope = parameters[0]
        intercept = parameters[1]
        
        if slope < 0:
            left_fit.append((slope, intercept))  # Eğimi negatifse sol şerit listesine ekliyoruz.
        else:
            right_fit.append((slope, intercept))  # Eğimi pozitifse sağ şerit listesine ekliyoruz.
    
    left_fit_average = np.average(left_fit, axis=0)  # Sol şerit çizgilerinin ortalamasını alıyoruz.
    right_fit_average = np.average(right_fit, axis=0)  # Sağ şerit çizgilerinin ortalamasını alıyoruz.
    
    left_lane = make_coordinates(image, left_fit_average)  # Sol şeridin koordinatlarını hesaplıyoruz.
    right_lane = make_coordinates(image, right_fit_average)  # Sağ şeridin koordinatlarını hesaplıyoruz.
    return np.array([left_lane, right_lane])  # Sol ve sağ şeritleri döndürüyoruz.

# Görüntüyü okuyup işlemleri başlatıyoruz.

image = cv2.imread("test_image.jpg")  # Test görüntüsünü yüklüyoruz.
lane_image = np.copy(image)  # Orijinal görüntüyü değiştirmemek için kopyasını alıyoruz.
canny_image = canny(lane_image)  # Kenar tespiti işlemini uyguluyoruz.
cropped_image = region_of_interest(canny_image)  # İlgi alanı maskesini uyguluyoruz.

# Hough dönüşümü ile çizgileri tespit ediyoruz.
lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
averaged_lines = average_slope_intercept(lane_image, lines)  # Tespit edilen çizgileri ortalama alarak düzenliyoruz.
line_image = display_lanes(lane_image, averaged_lines)  # Çizgileri çizdiriyoruz.

# Orijinal görüntüyle şerit çizgilerini birleştiriyoruz.
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

#cv2.imshow("Şerit Algılama", combo_image)  # Sonucu ekranda gösteriyoruz.
#cv2.waitKey(0)  # Görüntüyü kapatana kadar bekliyoruz.

# Alternatif olarak matplotlib ile de kenarları görselleştirebiliriz.
plt.imshow(canny_image)
plt.show()
