import cv2
import numpy as np
import time
import os
import math

def calculate_contour_metrics(contours):
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    
    # 計算原始輪廓的面積和圓度
    area_original = cv2.contourArea(cnt)
    perimeter_original = cv2.arcLength(cnt, True)
    circularity_original = (2 * math.sqrt(math.pi * area_original)) / perimeter_original if perimeter_original != 0 else 0
    
    # 計算凸包
    #hull = cv2.convexHull(cnt)
    
    # 計算凸包的面積和圓度
    #area_hull = cv2.contourArea(hull)
    #perimeter_hull = cv2.arcLength(hull, True)
    #circularity_hull = (2 * math.sqrt(math.pi * area_hull)) / perimeter_hull if perimeter_hull != 0 else 0
    
    # 計算比值
    #area_ratio = area_hull / area_original if area_original != 0 else 0
    #circularity_ratio = circularity_hull / circularity_original if circularity_original != 0 else 0

    results = {
        "area_original": area_original,
    #    "area_hull": area_hull,
    #    "area_ratio": area_ratio,
        "circularity_original": circularity_original,
    #    "circularity_hull": circularity_hull,
    #    "circularity_ratio": circularity_ratio,
        "contour": cnt,
    #    "hull": hull
    }

    return results

def process_image(image_path, background_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    background = cv2.imread(background_path, cv2.IMREAD_GRAYSCALE)
    if image is None or background is None:
        print(f"Error reading images from {image_path} or {background_path}")
        return
    
    blurred_bg = cv2.GaussianBlur(background, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    start_time = time.time()
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    bg_sub = cv2.subtract(blurred_bg, blurred)

    _, binary = cv2.threshold(bg_sub, 10, 255, cv2.THRESH_BINARY)

    dilate1 = cv2.dilate(binary, kernel, iterations=2)
    erode1 = cv2.erode(dilate1, kernel, iterations=2)
    erode2 = cv2.erode(erode1, kernel, iterations=1)
    dilate2 = cv2.dilate(erode2, kernel, iterations=1)

    edges = cv2.Canny(dilate2, 50, 150)

    contours = cv2.findContours(edges)
    metrics = calculate_contour_metrics(contours)

    if metrics:
        print("Original Area:", metrics["area_original"])
    #    print("Hull Area:", metrics["area_hull"])
    #    print("Area Ratio:", metrics["area_ratio"])
        print("Original Circularity:", metrics["circularity_original"])
    #    print("Hull Circularity:", metrics["circularity_hull"])
    #    print("Circularity Ratio:", metrics["circularity_ratio"])

    end_time = time.time()
    print("Processing time:", end_time - start_time)

# Replace 'path_to_image.tif' with your image file path
directory = 'Test_images/Test Slight under focus'
files = [f for f in os.listdir(directory) if f.endswith('.tiff')]
for image in files:
    image_path = os.path.join(directory, image)
    print(image_path)
    process_image(image_path, 'Test_images/Test Slight under focus/background.tiff')
