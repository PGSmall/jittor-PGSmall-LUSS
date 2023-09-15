python post-processing/Ensemble/ensemble.py --base_directory ./weights/pass50_r18_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz256/test \
--new_base_directory ./weights/pass50_r18_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz256/test_modify \
--r18_dir ./weights/pass50_r18_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz256/test_modify \
--r34_dir ./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/test \
--ensemble_dir ./weights/pass50_r34_bz128_ep400/pixel_finetuning_ep40_lr0.6_sz384/test_ensemble