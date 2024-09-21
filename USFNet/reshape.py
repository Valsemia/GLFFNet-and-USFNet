import os
import numpy as np
from skimage import io
from skimage.transform import resize
from PIL import Image

def resize_images_in_folder(folder_path, target_width, target_height):
    # 获取文件夹中的所有文件名
    file_names = os.listdir(folder_path)

    if not file_names:
        print("The folder is empty or the path is incorrect.")
        return

    # 遍历所有文件
    for file_name in file_names:
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否是图片格式（这里假设jpg、jpeg、png、bmp等常见图片格式）
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            print(f"Processing file: {file_name}")

            # 尝试使用 PIL 读取图片
            try:
                with Image.open(file_path) as img:
                    image = np.array(img)
            except Exception as e:
                print(f"Failed to read image using PIL: {file_path}, error: {e}")
                continue

            # 调整图片尺寸
            resized_image = resize(image, (target_height, target_width), preserve_range=True).astype(np.uint8)
            print(f"Image resized: {resized_image.shape}, dtype: {resized_image.dtype}")

            # 构建新的文件名和路径
            base_name, ext = os.path.splitext(file_name)
            new_file_name = f"{base_name}_reshape{ext}"
            new_file_path = os.path.join(folder_path, new_file_name)

            # 检查文件路径有效性
            if os.path.exists(new_file_path):
                print(f"File already exists: {new_file_path}")
                continue

            # 保存调整尺寸后的图片
            try:
                io.imsave(new_file_path, resized_image)
                print(f"Saved resized image to {new_file_path}")
            except Exception as e:
                print(f"Failed to save image: {new_file_path}, error: {e}")
        else:
            print(f"Skipping non-image file: {file_name}")

# 指定文件夹路径
folder_path = r"D:\Mine\研究生\实验室\Paper\小论文\去除结果图\对比图片\116-4"  # 替换为你的文件夹路径

# 目标尺寸
target_width = 640
target_height = 480

# 调整文件夹中所有图片的尺寸
resize_images_in_folder(folder_path, target_width, target_height)







