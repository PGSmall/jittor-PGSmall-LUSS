CUDA_VISIBLE_DEVICES=0 python post-processing/SAM-Step2/sam_pgs.py --img ./data/test \
--mask ./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/test_ensemble_unique \
--outdir ./result \
--ckpt ./post-processing/segment-anything-jittor/checkpoint/sam_vit_b_01ec64.pth
