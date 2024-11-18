import cv2
import numpy as np
import math
import os

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
for i in range(min(len(results_annotation), len(results_woannotation))):
    
    annotation_result = results_annotation[i]
    woannotation_result = results_woannotation[i]

    print(f"\nComparing Image Pair {i + 1}:")
    
    # 顯示各指標及其差異
    for key in ["area_original", "area_hull", "area_ratio", "circularity_original", "circularity_hull", "circularity_ratio"]:
        
        annotation_value = annotation_result[key]
        woannotation_value = woannotation_result[key]

        # 計算相對誤差百分比，以 WOannotation 為基準
        relative_error_percentage = ((annotation_value - woannotation_value) / woannotation_value * 100) if woannotation_value != 0 else float('inf')

        print(f"{key} (Annotation): {annotation_value}")
        print(f"{key} (WOannotation): {woannotation_value}")
        print(f"{key} Relative Error: {relative_error_percentage:.4f}%")
        print()