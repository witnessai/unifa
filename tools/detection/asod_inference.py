from mmdet.apis import init_detector, inference_detector
import mmcv
import ipdb 
import os 

# Specify the path to model config and checkpoint file
config_file = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd_73_80.py'
checkpoint_file = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = '/data1/niehui/MSCOCO/val2014/COCO_val2014_000000020650.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# ipdb.set_trace()
# print(result)

# visualize the results in a new window
score_thr = 0.05
model.show_result(img, result, score_thr)
# or save the visualization results to image files

save_root = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/inference_results/'
save_path = os.path.join(save_root, 'result.jpg')
model.show_result(img, result, out_file=save_path)



