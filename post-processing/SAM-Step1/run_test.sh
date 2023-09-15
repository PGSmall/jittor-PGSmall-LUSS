orignal_folder="/farm/peigensheng/USS/PASS-All/post-processing/CRF/test_crf"
source_folder="/farm/peigensheng/USS/PASS-All/post-processing/SAM_WSSS/data/test_pseudo"
target_folder="/farm/peigensheng/USS/PASS-All/post-processing/SAM_WSSS/data/test_masks_pseudo"
sam_folder="/farm/peigensheng/USS/PASS-All/post-processing/CRF/test_sam"

# step 1: move png test masks
cp -r  ${orignal_folder} ${source_folder}

mkdir -p ${target_folder}

find "$source_folder" -type d | while read -r dir; do
    find "$dir" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) -exec mv -t "$target_folder" {} +
done

# step2: refine masks by using SAM
processed_folder="./processed_mask"

if [ -d "$processed_folder" ]; then
    # This will remove the directory and its content.
    rm -rf "$processed_folder"
fi
python main.py --mode merge --pseudo_path ./data/test_masks_pseudo/ --sam_path ./data/test_masks_sam/

# step 3: copy sam masks

if [ -d "$sam_folder" ]; then
    # This will remove the directory and its content.
    rm -rf "$sam_folder"
fi

cp -r ${orignal_folder} ${sam_folder}

FILE="test_path.txt"

if [ -f "$FILE" ]; then
    # This will remove the file.
    rm "$FILE"
fi

find ${sam_folder} -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) > test_path.txt
python test_copy_masks.py
