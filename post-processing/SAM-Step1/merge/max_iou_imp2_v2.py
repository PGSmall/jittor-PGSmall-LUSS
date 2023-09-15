from merge.merge_base import Merger
import os
from PIL import Image
import numpy as np
from pathlib import Path

class MaxIoU_IMP2(Merger):
    def __init__(self, params, num_cls, threshold=0.2):
        super(MaxIoU_IMP2, self).__init__(params, num_cls)
        self.threshold = threshold

    def merge(self, predict, name, sam_folder, save_path):
        seen = []
        processed_mask = np.zeros_like(predict)
        # print("before class: ", np.unique(predict))
        for i in range(1, self.num_cls):
            pos_cls = predict == i
            if np.sum(pos_cls) == 0:
                continue
            iou = 0
            candidates = []
            sam_mask = np.zeros_like(pos_cls)
            neg_cls = predict == 0

            for filename in os.scandir(sam_folder):
                if filename.is_file() and filename.path.endswith('png') and filename.path not in seen:
                    cur_sam = np.array(Image.open(filename.path)) == 255
                    # print(filename.path)
                    sam_mask = np.logical_or(sam_mask, cur_sam)

                    neg_pred_thresh = np.sum((neg_cls == cur_sam) * neg_cls) / (np.sum(neg_cls) + np.finfo(np.float32).eps)
                    neg_sam_thresh = np.sum((neg_cls == cur_sam) * neg_cls) / (np.sum(cur_sam) + np.finfo(np.float32).eps)

                    pos_thresh = 2 * np.sum((pos_cls == cur_sam) * pos_cls) - np.sum(cur_sam)
                    pos_pred_thresh = np.sum((pos_cls == cur_sam) * pos_cls) / np.sum(pos_cls)

                    if neg_pred_thresh + neg_sam_thresh < 0.88:
                        if pos_thresh > 0.5 or pos_pred_thresh >= 0.85:
                            candidates.append(cur_sam)
                            seen.append(filename.path)
                            iou += np.sum(pos_cls == cur_sam)
                    
            cam_mask = np.logical_and(sam_mask==0, pos_cls==1)
            # Trust CAM if SAM has no prediction on that pixel
            candidates.append(cam_mask)
            processed_mask[np.sum(candidates, axis=0) > 0] = i
        
        # im = Image.fromarray(processed_mask)
        mask = np.zeros((processed_mask.shape[0], processed_mask.shape[1], 3))
        mask[:, :, 0] = processed_mask % 256
        mask[:, :, 1] = processed_mask // 256
        # print("after class: ", np.unique(mask[:, :, 0]))
        im = Image.fromarray(mask.astype(np.uint8))
        im.save(f'{save_path}/{name}.png')