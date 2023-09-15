import os
import shutil
from os import listdir
from os.path import join
from tqdm import tqdm

import argparse
import numpy as np
import pydensecrf.densecrf as dcrf
from multiprocessing import Pool

try:
    from cv2 import imread, imwrite
except ImportError:
    # 如果没有安装OpenCV，就是用skimage
    from skimage.io import imread, imsave
    imwrite = imsave

from PIL import Image

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


def CRFs(gt_prob, original_image_path,predicted_image_path,CRF_image_path):
    img = imread(original_image_path)
    anno_rgb = imread(predicted_image_path).astype(np.uint32)

    if img.shape != anno_rgb.shape:
        print("Inconsistent image and label sizes!")
        print(original_image_path)
        anno_rgb = anno_rgb.transpose((1, 0, 2))
    
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    colors, labels = np.unique(anno_lbl, return_inverse=True)
    colorize = np.zeros((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    n_labels = len(set(labels.flat))
    
    d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

    try:
        U = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=None)
        d.setUnaryEnergy(U)
    
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=2,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
    
        feats = create_pairwise_bilateral(sdims=(30, 30), schan=(7, 7, 7),
                                        img=img, chdim=2)

        d.addPairwiseEnergy(feats, compat=12,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(20)
        MAP = np.argmax(Q, axis=0)
        
        MAP = colorize[MAP,:]
        MAP = MAP.reshape(img.shape)
        MAP = MAP[:, :, ::-1]
        res = Image.fromarray(MAP.astype(np.uint8))
        res.save(CRF_image_path)

    except ZeroDivisionError:
            shutil.copy(predicted_image_path, CRF_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/test')
    parser.add_argument('--mask_dir', type=str, default='./data/test_mask')
    parser.add_argument('--save_dir', type=str, default='./data/test_crf')
    parser.add_argument('--gt_prob', type=float, default=0.9)
    args = parser.parse_args()
    
    image_dir = args.image_dir
    mask_dir = args.mask_dir
    save_dir = args.save_dir
    gt_prob = args.gt_prob

    for seq in tqdm(listdir(mask_dir)):
        if os.path.isdir(join(mask_dir, seq)):
            seq_dir = join(image_dir, seq)
            seq_mask_dir = join(mask_dir, seq)
            res_dir = join(save_dir, seq)
            if not os.path.exists(res_dir):
                os.makedirs(res_dir)
            
            results = []
            with Pool(processes=16) as pool:
                for f in listdir(seq_mask_dir):
                    img_path = join(seq_dir, f[:-4] + '.JPEG')
                    mask_path = join(seq_mask_dir, f)
                    save_path = join(res_dir, f)
                    pool.apply_async(CRFs, (gt_prob, img_path , mask_path, save_path))
                
                pool.close()
                pool.join()
    # shutil.make_archive(save_dir, 'zip', root_dir=save_dir)