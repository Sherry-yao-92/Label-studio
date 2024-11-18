import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy import stats

def calculate_contour_metrics(contours):
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    
    # 計算原始輪廓的面積和圓度
    area_original = cv2.contourArea(cnt)
    perimeter_original = cv2.arcLength(cnt, True)
    circularity_original = float(2 * math.sqrt((math.pi) * area_original)) / perimeter_original
    
    # 計算凸包
    hull = cv2.convexHull(cnt)
    
    # 計算凸包的面積和圓度
    area_hull = cv2.contourArea(hull)
    perimeter_hull = cv2.arcLength(hull, True)
    circularity_hull = float(2 * math.sqrt((math.pi) * area_hull)) / perimeter_hull
    
    # 計算比值
    area_ratio = area_hull / area_original if area_original != 0 else 0
    circularity_ratio = circularity_hull / circularity_original if circularity_original != 0 else 0

    results = {
        "area_original": area_original,
        "area_hull": area_hull,
        "area_ratio": area_ratio,
        "circularity_original": circularity_original,
        "circularity_hull": circularity_hull,
        "circularity_ratio": circularity_ratio,
        "contour": cnt,
        "hull": hull
    }

    return results

def annotation(img_path1):
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)

    if img1 is None:
        print(f"Error: Unable to read image {img_path1}")
        return None

    contours, _ = cv2.findContours(img1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    results1 = calculate_contour_metrics(contours)

    return results1

def WOannotation(img_path2, background_path):
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)

    if img2 is None or background is None:
        print(f"Error reading images from {img_path2} or {background_path}")
        return None
    
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    blurred = cv2.GaussianBlur(img2, (5, 5), 0)
    bg_sub = cv2.subtract(blurred_bg, blurred)

    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)

    dilate1 = cv2.dilate(binary, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)
    dilate2 = cv2.dilate(erode2, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilate2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    results2 = calculate_contour_metrics(contours)

    return results2

# 主程序
if __name__ == "__main__":
    img_folder1 = 'annotation_single_cell'
    img_folder2 = 'Test_images/Test Slight under focus'
    
    background_path = 'Test_images/Test Slight under focus/background.tiff'  # 替換為你的背景圖片路徑

    results_annotation = []
    results_woannotation = []

    for filename in os.listdir(img_folder1):
        if filename.endswith('.png'):
            img_path1 = os.path.join(img_folder1, filename)
            results1 = annotation(img_path1)

            if results1:
                results_annotation.append(results1)

    for filename in os.listdir(img_folder2):
        if filename.endswith('.tiff'):
            img_path2 = os.path.join(img_folder2, filename)
            results2 = WOannotation(img_path2, background_path)

            if results2:
                results_woannotation.append(results2)

# 比較兩組結果（假設兩組圖片數量相同）
keys_to_compare = ["area_original", "area_hull", "area_ratio", "circularity_original", "circularity_hull", "circularity_ratio"]

for key in keys_to_compare:
    
   annotation_values = [result[key] for result in results_annotation]
   woannotation_values = [result[key] for result in results_woannotation]

   # 計算平均值和 p 值
   mean_annotation = np.mean(annotation_values)
   mean_woannotation = np.mean(woannotation_values)
   t_statistic, p_value_temp = stats.ttest_ind(annotation_values, woannotation_values)

   # 繪製誤差條圖，顯示整體比較結果
   plt.figure(figsize=(8, 5))
   x_labels = ['Annotation', 'WOannotation']
   means = [mean_annotation, mean_woannotation]
   stds = [np.std(annotation_values), np.std(woannotation_values)]

   x_pos = np.arange(len(x_labels))

   bars = plt.bar(x_pos, means, yerr=stds, capsize=5)

   # 在條形上標示 p 值
   plt.text(0.5 , max(means) + 0.05,
            f'p={p_value_temp:.3f}', ha='center', va='bottom')

   plt.title(f'Comparison of {key} with P-Value')
   plt.ylabel('Mean Value')
   plt.xticks(x_pos, x_labels)
   plt.ylim(0, max(means) + max(stds) + 1)  # 設定 y 軸範圍以顯示誤差條
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   plt.tight_layout()
   plt.show()