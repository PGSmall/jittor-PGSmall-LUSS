orignal_folder="./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/test_ensemble_crf"
source_folder="./post-processing/SAM-Step1/data/test_pseudo"
target_folder="./post-processing/SAM-Step1/data/test_masks_pseudo"
sam_folder="./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/test_ensemble_sam"

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
python ./post-processing/SAM-Step1/main.py --mode merge --pseudo_path ./post-processing/SAM-Step1/data/test_masks_pseudo/ --sam_path ./data/test_sam/

# step 3: copy sam masks

if [ -d "$sam_folder" ]; then
    # This will remove the directory and its content.
    rm -rf "$sam_folder"
fi

cp -r ${orignal_folder} ${sam_folder}

FILE="./post-processing/SAM-Step1/test_path.txt"

if [ -f "$FILE" ]; then
    # This will remove the file.
    rm "$FILE"
fi

find ${sam_folder} -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) > ./post-processing/SAM-Step1/test_path.txt
python ./post-processing/SAM-Step1/test_copy_masks.py
