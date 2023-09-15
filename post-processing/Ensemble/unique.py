import os
import json
import argparse

from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('--mask_dir', type=str, default='./test_ensemble_sam')
parser.add_argument('--unique_dir', type=str, default='./test_unique')
args = parser.parse_args()

# 打开json文件
json_r34_path = os.path.join('./post-processing/Ensemble', 'match_r34.json')
with open(json_r34_path) as f:
    data_r34 = json.load(f)

# 打开txt文件
txt_path = os.path.join('./post-processing/Ensemble', 'test_unique.txt')
with open(txt_path, 'r') as f:
    lines = f.readlines()

# 定义一个函数，根据value获取json文件中的对应key
def get_key(data, value):
    for k, v in data.items():
        if str(v) == value:
            return int(k)
    return "There is no such value."

for line in lines:
    split_line = line.strip().split()
    if split_line[3] == 'test':
        index = split_line[0]
        directory = split_line[1]
        new_value = get_key(data_r34, index)
        
        path = os.path.join(args.mask_dir, directory)
        save_path = os.path.join(args.unique_dir, directory)
        os.makedirs(save_path, exist_ok=True)
        for file_name in os.listdir(path):
            # 检查是否是png文件
            if file_name.endswith('.png'):
                img_path = os.path.join(path, file_name)
                save_img_path = os.path.join(save_path, file_name)
                
                # 加载图像并获取R通道
                img = Image.open(img_path).convert('RGB')
                r, g, b = img.split()
                
                # 创建一个新的R通道图像并用新值填充非0的位置
                r_new = r.point(lambda p: new_value if p > 0 else p)

                print(file_name, new_value)
                # 合并回原色彩空间并保存
                img_new = Image.merge('RGB', (r_new, g, b))
                img_new.save(save_img_path)
