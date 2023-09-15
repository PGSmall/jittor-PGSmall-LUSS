import argparse

import os
from PIL import Image


r18_categories = {
    "1": 25, "2": 16, "3": 37, "4": 44, "5": 17, "6": 22, "7": 7, "8": 3,
    "9": 36, "10": 23, "11": 40, "12": 12, "13": 8, "14": 15, "15": 39,
    "16": 32, "17": 24, "18": 20, "19": 33, "20": 6, "21": 48, "22": 42,
    "23": 26, "24": 38, "25": 47, "26": 4, "27": 14, "28": 49, "29": 28,
    "30": 50, "31": 1, "32": 45, "33": 29, "34": 30, "35": 9, "36": 27,
    "37": 11, "38": 19, "39": 41, "40": 18, "41": 21, "42": 2, "43": 31,
    "44": 46, "45": 10, "46": 35, "47": 43, "48": 5, "49": 13, "50": 34,
    "0": 0
}

r34_categories = {
    "1": 27, "2": 17, "3": 18, "4": 35, "5": 19, "6": 50, "7": 11, "8": 28,
    "9": 6, "10": 20, "11": 36, "12": 39, "13": 46, "14": 47, "15": 15,
    "16": 12, "17": 33, "18": 2, "19": 44, "20": 41, "21": 49, "22": 43,
    "23": 45, "24": 42, "25": 21, "26": 9, "27": 10, "28": 34, "29": 13,
    "30": 7, "31": 26, "32": 25, "33": 14, "34": 1, "35": 37, "36": 8,
    "37": 5, "38": 29, "39": 16, "40": 23, "41": 4, "42": 40, "43": 32,
    "44": 48, "45": 22, "46": 38, "47": 24, "48": 31, "49": 3, "50": 30,
    "0": 0
}

category_mapping = {}  # 存储转换后的r18 mask结果

for r18_category_id, folder_number in r18_categories.items():
    if folder_number in r34_categories.values():
        r34_category_id = next(key for key, value in r34_categories.items() if value == folder_number)
        category_mapping[r34_category_id] = r18_category_id

# 颠倒key和value
category_mapping = {v: k for k, v in category_mapping.items()}

# print(category_mapping)

parser = argparse.ArgumentParser()
parser.add_argument('--base_directory', type=str, default='./r18_ori')
parser.add_argument('--new_base_directory', type=str, default='./r18_modify')
parser.add_argument('--r18_dir', type=str, default='./r18_modify')
parser.add_argument('--r34_dir', type=str, default='./r34_ori')
parser.add_argument('--ensemble_dir', type=str, default='./test_ensemble')
args = parser.parse_args()

# 指定原目录路径和新目录路径
# base_directory = "./r18_ori"
# new_base_directory = "./r18_modify"
base_directory = args.base_directory
new_base_directory = args.new_base_directory

# 遍历子目录
for subdir in os.listdir(base_directory):
    subdir_path = os.path.join(base_directory, subdir)
    if os.path.isdir(subdir_path):
        new_subdir_path = os.path.join(new_base_directory, subdir)
        os.makedirs(new_subdir_path, exist_ok=True)

        # 遍历子目录中的PNG文件
        for filename in os.listdir(subdir_path):
            if filename.endswith(".png"):
                file_path = os.path.join(subdir_path, filename)
                new_file_path = os.path.join(new_subdir_path, filename)

                # 打开PNG文件并修改类别
                image = Image.open(file_path)
                r, g, b = image.split()
                pixels = r.load()
                for y in range(r.height):
                    for x in range(r.width):
                        pixel_value = pixels[x, y]
                        new_value = category_mapping.get(str(pixel_value), pixel_value)
                        pixels[x, y] = int(new_value)

                # 合并通道并保存修改后的图像
                image = Image.merge("RGB", (r, g, b))
                image.save(new_file_path)
                print(f"Modified and saved: {new_file_path}")
######################################################################################


# 打开txt文件
txt_path = os.path.join('./post-processing/Ensemble', 'val_fscore.txt')
with open(txt_path, 'r') as f:
    lines = f.readlines()


for line in lines:
    split_line = line.strip().split()
    if split_line[3] == 'r18':
        index = split_line[0]
        directory = split_line[1]
        
        path = os.path.join(args.r18_dir, directory)
        save_path = os.path.join(args.ensemble_dir, directory)
        # 新建一个文件夹用于保存选择的子目录
        os.makedirs(save_path, exist_ok=True)
        for file_name in os.listdir(path):
            # 检查是否是png文件
            if file_name.endswith('.png'):
                img_path = os.path.join(path, file_name)
                save_img_path = os.path.join(save_path, file_name)
                
                # 加载图像并获取R通道
                img = Image.open(img_path).convert('RGB')
                img.save(save_img_path)
    
    elif split_line[3] == 'r34':
        index = split_line[0]
        directory = split_line[1]

        path = os.path.join(args.r34_dir, directory)
        save_path = os.path.join(args.ensemble_dir, directory)
        # 新建一个文件夹用于保存选择的子目录
        os.makedirs(save_path, exist_ok=True)
        for file_name in os.listdir(path):
            # 检查是否是png文件
            if file_name.endswith('.png'):
                img_path = os.path.join(path, file_name)
                save_img_path = os.path.join(save_path, file_name)
                
                # 加载图像并获取R通道
                img = Image.open(img_path).convert('RGB')
                img.save(save_img_path)
