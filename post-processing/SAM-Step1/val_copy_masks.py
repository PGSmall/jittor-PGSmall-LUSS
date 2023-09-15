import os
import shutil

source_directory = "./processed_mask"
address_file = "val_path.txt"

# 逐行读取地址文件
with open(address_file, "r") as file:
    for line in file:
        image_address = line.strip()
        image_file = os.path.basename(image_address)
        image_parent_directory = os.path.dirname(image_address)
        image_path = os.path.join(source_directory, image_file)
        # 检查是否为PNG图像文件
        if image_file.lower().endswith(".png"):
            # 检查图像文件是否存在
            if os.path.isfile(image_path):
                # 复制图像文件到目标目录
                shutil.copy(image_path, os.path.join(image_parent_directory, image_file))
                # print(f"Copied {image_file} to {os.path.join(image_parent_directory, image_file)}")
            else:
                print(f"File {image_file} does not exist in {source_directory}")
        else:
            print(f"File {image_file} is not a PNG image")
