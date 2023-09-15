CUDA='0'
IMAGENETS=./data

DUMP_PATH=./weights/pass50_r34_bz128_ep400
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning_ep40_lr0.6_sz384

ARCH=resnet34
NUM_CLASSES=50

CUDA_VISIBLE_DEVICES=${CUDA} python inference_384.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode test \
--match_file ${DUMP_PATH_SEG}/validation/match.json
