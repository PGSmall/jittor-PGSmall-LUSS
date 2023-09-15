# 训练resnet18模型
# 在"./src/resnet.py"中，取消L146-165注释，并注释L168-202，采用Mulpixelattn(原始)
bash ./scripts/pass_r18.sh

# 训练resnet34模型
# 在"./src/resnet.py"中，取消L168-202注释，并注释L146-165，采用Mulpixelattn(增强)
bash ./scripts/pass_r34.sh