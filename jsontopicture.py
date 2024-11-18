import json
import numpy as np
from PIL import Image, ImageDraw

# 讀取 JSON 標註檔案
with open('Labelstudio.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


# 假設每個標註對象中都有 "imageWidth" 和 "imageHeight" 的欄位來表示圖片的寬高
image_width = data[0].get('imageWidth', 256)  # 如果沒有，則使用預設 256
image_height = data[0].get('imageHeight', 256)  # 如果沒有，則使用預設 256

# 創建一個與原始圖片大小一致的全黑圖片 (二值化圖片: 背景是黑色)
binary_image = np.zeros((image_height, image_width), dtype=np.uint8)

# 遍歷所有標註資料，選擇標籤為 'single cell' 的標註
for annotation in data:
    if annotation.get('label') == 'single cell':
        # 獲取該標註的點座標，格式是 [[x1, y1], [x2, y2], ...]
        points = annotation.get('points', [])
        
        if points:
            # 使用 PIL 的 ImageDraw 畫圖工具來畫多邊形
            pil_image = Image.fromarray(binary_image * 255)  # 創建 PIL 物件
            draw = ImageDraw.Draw(pil_image)
            
            # 將點組成的多邊形畫出來並填充白色
            draw.polygon(points, outline=255, fill=255)

            # 更新二值圖片
            binary_image = np.array(pil_image) // 255  # 將圖片轉換為二值格式 (0 或 1)

# 將二值化圖片轉換為 PIL 影像並顯示
binary_image_pil = Image.fromarray(binary_image * 255)  # 將 0 變為黑，1 變為白
binary_image_pil.show()

# 儲存二值化圖片
binary_image_pil.save('binary_image_single_cell.png')
