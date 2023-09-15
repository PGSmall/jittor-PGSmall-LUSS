CUDA='0'
N_GPU=1
BATCH=128
DATA=./data
IMAGENETS=./data

DUMP_PATH=./weights/pass50_r34_bz128_ep400
DUMP_PATH_FINETUNE=${DUMP_PATH}/pixel_attention_ep100_lr0.01
DUMP_PATH_SEG=${DUMP_PATH}/pixel_finetuning_ep40_lr0.6_sz384
QUEUE_LENGTH=2048
QUEUE_LENGTH_PIXELATT=3840
HIDDEN_DIM=512
NUM_PROTOTYPE=500
ARCH=resnet34
NUM_CLASSES=50
EPOCH=400
EPOCH_PIXELATT=100
EPOCH_SEG=40
FREEZE_PROTOTYPES=1001
FREEZE_PROTOTYPES_PIXELATT=0

mkdir -p ${DUMP_PATH_FINETUNE}
mkdir -p ${DUMP_PATH_SEG}

CUDA_VISIBLE_DEVICES=${CUDA} python main_pretrain.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH} \
--nmb_crops 2  \
--size_crops 224  \
--min_scale_crops 0.08  \
--max_scale_crops 1.  \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH} \
--epoch_queue_starts 15 \
--epochs ${EPOCH} \
--batch_size ${BATCH} \
--optim sgd \
--base_lr 0.6 \
--final_lr 0.0006  \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES} \
--wd 0.000001 \
--warmup_epochs 30 \
--workers 8 \
--seed 1208 \
--shallow 3 \
--weights 1 1

CUDA_VISIBLE_DEVICES=${CUDA} python main_pixel_attention.py \
--arch ${ARCH} \
--data_path ${IMAGENETS}/train \
--dump_path ${DUMP_PATH_FINETUNE} \
--nmb_crops 2 \
--size_crops 224 \
--min_scale_crops 0.08 \
--max_scale_crops 1. \
--crops_for_assign 0 1 \
--temperature 0.1 \
--epsilon 0.05 \
--sinkhorn_iterations 3 \
--feat_dim 128 \
--hidden_mlp ${HIDDEN_DIM} \
--nmb_prototypes ${NUM_PROTOTYPE} \
--queue_length ${QUEUE_LENGTH_PIXELATT} \
--epoch_queue_starts 0 \
--epochs ${EPOCH_PIXELATT} \
--batch_size ${BATCH} \
--optim sgd \
--base_lr 0.01 \
--final_lr 0.0006 \
--freeze_prototypes_niters ${FREEZE_PROTOTYPES_PIXELATT} \
--wd 0.000001 \
--warmup_epochs 0 \
--workers 8 \
--seed 1208 \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=${CUDA} python cluster.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES}

### Evaluating the pseudo labels on the validation set.
CUDA_VISIBLE_DEVICES=${CUDA} python inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode validation \
--test \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy

CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_FINETUNE} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation \
--curve \
--min 25 \
--max 70


CUDA_VISIBLE_DEVICES=${CUDA} python inference_pixel_attention.py -a ${ARCH} \
--pretrained ${DUMP_PATH_FINETUNE}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_FINETUNE} \
-c ${NUM_CLASSES} \
--mode train \
--centroid ${DUMP_PATH_FINETUNE}/cluster/centroids.npy \
-t 0.47

# t 0.47 mIoU 34.70; t 0.48 mIoU 34.71

CUDA_VISIBLE_DEVICES=${CUDA} python main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--base_lr 0.01 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--checkpoint_freq 2 \
--workers 8 \
--seed 1208 \
--num_classes ${NUM_CLASSES} \
--pseudo_path ${DUMP_PATH_FINETUNE}/train \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar

CUDA_VISIBLE_DEVICES=${CUDA} python inference_384.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode validation \
--match_file ${DUMP_PATH_SEG}/validation/match.json

CUDA_VISIBLE_DEVICES=${CUDA} python evaluator.py \
--predict_path ${DUMP_PATH_SEG} \
--data_path ${IMAGENETS} \
-c ${NUM_CLASSES} \
--mode validation

CUDA_VISIBLE_DEVICES=${CUDA} python inference_384.py -a ${ARCH} \
--pretrained ${DUMP_PATH_SEG}/checkpoint.pth.tar \
--data_path ${IMAGENETS} \
--dump_path ${DUMP_PATH_SEG} \
-c ${NUM_CLASSES} \
--mode train \
--match_file ${DUMP_PATH_SEG}/validation/match.json

# crf 0.6 for train
python post-processing/CRF/fast_densecrf.py --image_dir ./data/train \
--mask_dir ./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/train \
--save_dir ./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/train_crf
--gt_prob 0.6

# sam pseudo (exist)
# bash ./post-processing/segment-anything-jittor/train.sh

# sam refine
orignal_folder="./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/train_crf"
source_folder="./post-processing/SAM-Step1/data/train_pseudo"
target_folder="./post-processing/SAM-Step1/data/train_masks_pseudo"
sam_folder="./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/train_sam"

# step 1: move png test masks
cp -r  ${orignal_folder} ${source_folder}

mkdir -p ${target_folder}

find "$source_folder" -type d | while read -r dir; do
    find "$dir" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) -exec mv -t "$target_folder" {} +
done

# step2: refine masks by using SAM
processed_folder="./post-processing/SAM-Step1/processed_mask"

if [ -d "$processed_folder" ]; then
    # This will remove the directory and its content.
    rm -rf "$processed_folder"
fi
python ./post-processing/SAM-Step1/main.py --mode merge --pseudo_path ./post-processing/SAM-Step1/data/train_masks_pseudo/ --sam_path ./data/train_sam/

# step 3: copy sam masks

if [ -d "$sam_folder" ]; then
    # This will remove the directory and its content.
    rm -rf "$sam_folder"
fi

cp -r ${orignal_folder} ${sam_folder}

FILE="./post-processing/SAM-Step1/train_path.txt"

if [ -f "$FILE" ]; then
    # This will remove the file.
    rm "$FILE"
fi

find ${sam_folder} -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) > ./post-processing/SAM-Step1/train_path.txt
python ./post-processing/SAM-Step1/train_copy_masks.py

# retrain
CUDA_VISIBLE_DEVICES=${CUDA} python main_pixel_finetuning.py \
--arch ${ARCH} \
--data_path ${DATA}/train \
--dump_path ${DUMP_PATH_SEG} \
--epochs ${EPOCH_SEG} \
--batch_size ${BATCH} \
--img_size 384 \
--base_lr 0.6 \
--final_lr 0.0006 \
--wd 0.000001 \
--warmup_epochs 0 \
--checkpoint_freq 2 \
--workers 8 \
--seed 1208 \
--num_classes ${NUM_CLASSES} \
--pseudo_path ${DUMP_PATH_SEG}/train_sam \
--pretrained ${DUMP_PATH}/checkpoint.pth.tar