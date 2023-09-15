import numpy as np
# import torch
# from torch.nn import functional as F
import jittor as jt
from jittor import nn
jt.flags.use_cuda = 1

import os
import cv2
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')

# from show import *
# from per_segment_anything import sam_model_registry, SamPredictor
from per_segment_anything_jittor import sam_model_registry, SamPredictor



def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', type=str, default='./data/test')
    parser.add_argument('--mask', type=str, default='./mask')
    parser.add_argument('--outdir', type=str, default='./result')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_b_01ec64.pth')

    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)

    images_path = args.img
    masks_path = args.mask
    output_path = args.outdir

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    print("======> Load SAM" )
    sam_type, sam_ckpt = 'vit_b', args.ckpt
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    
    for obj_name in os.listdir(images_path):
        if ".DS" not in obj_name:
            persam(args, sam, obj_name, images_path, masks_path, output_path)
        jt.sync_all()
        jt.gc()


def persam(args, sam, obj_name, images_path, masks_path, output_path):

    print("\n------------> Segment " + obj_name)
    
    # Path preparation
    imgs_list = os.listdir(os.path.join(images_path, obj_name))
    for i in range(0, len(imgs_list)):
        ref_idx = imgs_list[i][:-5]
        print(ref_idx)
        ref_image_path = os.path.join(images_path, obj_name, ref_idx + '.JPEG')
        ref_mask_path = os.path.join(masks_path, obj_name, ref_idx + '.png')
        test_images_path = os.path.join(images_path, obj_name)
        save_path = os.path.join(output_path, obj_name)
        os.makedirs(save_path, exist_ok=True)

        # Load images and masks
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        ref_mask_temp = cv2.imread(ref_mask_path)
        if ref_mask_temp is None:
            continue
        else:
            test_idx = imgs_list[i][:-5]
            mask_output_path = os.path.join(save_path, test_idx + '.png')
            ref_mask_temp = cv2.cvtColor(ref_mask_temp, cv2.COLOR_BGR2RGB)
            class_ids = np.unique(ref_mask_temp[:, :, 0])

            if len(class_ids) == 1:
                Image.fromarray(ref_mask_temp).save(mask_output_path)
                continue

            else:
                mask_colors = np.zeros((ref_mask_temp.shape[0], ref_mask_temp.shape[1], 3)).astype(np.uint8)
                for id in class_ids[1:]:
                    print(id)
                    ref_mask = np.zeros((ref_mask_temp.shape[0], ref_mask_temp.shape[1], 3)).astype(np.uint8)
                    mask = np.select([ref_mask_temp[:, :, 0] == id], [ref_mask_temp[:, :, 0] + 200], default=0)
                    
                    pos_mask = sum(sum(mask != 0)) / (mask.shape[0] * mask.shape[1])
                    if pos_mask < 0.1:
                        Image.fromarray(ref_mask_temp).save(mask_output_path)
                        continue
                    else:
                        # ref_mask[:, :, 0] = torch.from_numpy(mask)
                        ref_mask = jt.array(ref_mask)
                        ref_mask[:, :, 0] = jt.array(mask)

                        print("======> Obtain Location Prior" )
                        predictor = SamPredictor(sam)

                        # Image features encoding
                        ref_mask = predictor.set_image(ref_image, ref_mask)
                        ref_feat = predictor.features.squeeze().permute(1, 2, 0)

                        # ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
                        ref_mask = nn.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
                        ref_mask = ref_mask.squeeze()[0]

                        # Target feature extraction
                        target_feat = ref_feat[ref_mask > 0]
                        target_embedding = target_feat.mean(0).unsqueeze(0)
                        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
                        target_embedding = target_embedding.unsqueeze(0)


                        print('======> Start Testing')
                        
                        # Load test image
                        test_image_path = test_images_path + '/' + test_idx + '.JPEG'
                        test_image = cv2.imread(test_image_path)
                        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

                        # Image feature encoding
                        predictor.set_image(test_image)
                        test_feat = predictor.features.squeeze()

                        # Cosine similarity
                        C, h, w = test_feat.shape
                        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
                        test_feat = test_feat.reshape(C, h * w)
                        sim = target_feat @ test_feat
                        # probs = (100.0 * sim).softmax(dim=-1)
                        # if torch.sum(sim)/4096 < 0.8:
                        #     Image.fromarray(ref_mask_temp).save(mask_output_path)
                        #     continue
                        # else:
                        sim = sim.reshape(1, 1, h, w)
                        # sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
                        sim = nn.interpolate(sim, scale_factor=4, mode="bilinear")
                        sim = predictor.model.postprocess_masks(
                                        sim,
                                        input_size=predictor.input_size,
                                        original_size=predictor.original_size).squeeze()

                        # Positive-negative location prior
                        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
                        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
                        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

                        # Obtain the target guidance for cross-attention layers
                        # sim = (sim - sim.mean()) / torch.std(sim)
                        sim = (sim - sim.mean()) / jt.std(sim)
                        # sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
                        sim = nn.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
                        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)
                        # print("attn_sim:", attn_sim.shape)

                        # First-step prediction
                        masks, scores, logits, _ = predictor.predict(
                            point_coords=topk_xy, 
                            point_labels=topk_label, 
                            multimask_output=False,
                            attn_sim=attn_sim,  # Target-guided Attention
                            target_embedding=target_embedding  # Target-semantic Prompting
                        )
                        best_idx = 0

                        # Cascaded Post-refinement-1
                        masks, scores, logits, _ = predictor.predict(
                                    point_coords=topk_xy,
                                    point_labels=topk_label,
                                    mask_input=logits[best_idx: best_idx + 1, :, :], 
                                    multimask_output=True)
                        best_idx = np.argmax(scores)

                        # Cascaded Post-refinement-2
                        y, x = np.nonzero(masks[best_idx])
                        if len(y) == 0 or len(x) == 0:
                            Image.fromarray(ref_mask_temp).save(mask_output_path)
                            continue
                        else:
                            x_min = x.min()
                            x_max = x.max()
                            y_min = y.min()
                            y_max = y.max()
                            input_box = np.array([x_min, y_min, x_max, y_max])
                            masks, scores, logits, _ = predictor.predict(
                                point_coords=topk_xy,
                                point_labels=topk_label,
                                box=input_box[None, :],
                                mask_input=logits[best_idx: best_idx + 1, :, :], 
                                multimask_output=True)
                            best_idx = np.argmax(scores)
                            
                            # print("scores: ", scores)
                            # print("best_idx: ", best_idx)
                            # print("best score: ", scores[best_idx])
                            # if scores[best_idx] < 0.95:
                            #     Image.fromarray(ref_mask_temp).save(mask_output_path)
                            #     continue
                            # else:
                            #     # Save masks
                            final_mask = masks[best_idx]

                            pre_masks = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
                            pre_masks[final_mask, :] = np.array([[200, 0, 0]], dtype=np.uint8)
                            # mask_output_path = os.path.join(save_path, test_idx + '.png')
                            # cv2.imwrite(mask_output_path, mask_colors)
                        
                        # Pre Image features encoding
                        # temp_mask = predictor.set_image(test_image, pre_masks)
                        # temp_feat = predictor.features.squeeze().permute(1, 2, 0)

                        # temp_mask = F.interpolate(temp_mask, size=temp_feat.shape[0: 2], mode="bilinear")
                        # temp_mask = temp_mask.squeeze()[0]

                        # # Pre feature extraction
                        # pre_feat = temp_feat[temp_mask > 0]
                        # pre_embedding = pre_feat.mean(0).unsqueeze(0)
                        # pre_feat = pre_embedding / pre_embedding.norm(dim=-1, keepdim=True)

                        # # print(F.cosine_similarity(target_feat, pre_feat))
                        # # sim = target_feat @ pre_feat
                        # sim = F.cosine_similarity(target_feat, pre_feat)
                        # print("cosine_similarity:", sim)
                        # print(mask.shape, pre_masks[..., 0].shape)
                        # print(mask, pre_masks[..., 0])

                        # iou = mask_iou(torch.from_numpy(mask).float(), torch.from_numpy(pre_masks[..., 0]).float())
                        # mask[mask > 0] = 1
                        # pre_mask = pre_masks[..., 0]
                        # pre_mask[pre_mask > 0] = 1
                        # iou = mask_iou(torch.from_numpy(mask).float(), torch.from_numpy(pre_mask).float())
                        # sim_iou = torch.sum(iou>0)/(mask.shape[0] * mask.shape[1])
                        sim_iou = mask_iou(mask, pre_masks[..., 0])
                        print(sim_iou)
                        if  sim_iou < 0.7:
                            Image.fromarray(ref_mask_temp).save(mask_output_path)
                            continue
                        else:
                        # pre_feat = test_feat[pre_masks > 0]
                        # pre_embedding = pre_feat.mean(0).unsqueeze(0)
                        # pre_feat = pre_embedding / pre_embedding.norm(dim=-1, keepdim=True)

                            mask_colors[final_mask, :] = np.array([[id, 0, 0]], dtype=np.uint8)
                # print("class ids: ", np.unique(mask_colors))
                # print("class len: ", len(np.unique(mask_colors)))
                if len(np.unique(mask_colors)) == 1:
                    Image.fromarray(ref_mask_temp).save(mask_output_path)
                    continue
                else:
                    Image.fromarray(mask_colors).save(mask_output_path)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    # topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_xy = jt.contrib.concat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
        
    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    # last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_xy = jt.contrib.concat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    
    return topk_xy, topk_label, last_xy, last_label


def mask_iou(mask1, mask2):
    """
    mask1: [m1,n] m1 means number of predicted objects 
    mask2: [m2,n] m2 means number of gt objects
    Note: n means image_w x image_h
    """
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    # print(intersection)
    union = np.sum((mask1 > 0) | (mask2 > 0))
    iou = intersection / union
    return iou


if __name__ == "__main__":
    main()
