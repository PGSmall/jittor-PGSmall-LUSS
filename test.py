import os

# 使用SAM推理测试集，生成候选masks，所需显存较大
# 我们已将推理结果已保存在data/test_sam目录下，也可以取消下面这行注释来产生，结果一致。
# os.system('bash ./post-processing/segment-anything-jittor/run.sh')

# 集成resnet18和resnet34模型
os.system('bash ./scripts/test_r18.sh')
os.system('bash ./scripts/test_r34.sh')
os.system('bash ./scripts/test_ensemble.sh')

# 后处理
os.system('bash ./scripts/test_crf.sh')     # crf
os.system('bash ./scripts/test_sam.sh')     # sam refine
os.system('bash ./scripts/test_unique.sh')  # sam unique
os.system('bash ./scripts/test_persam.sh')  # persam 