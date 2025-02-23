import ipdb 
import shutil
import os 


### copy from train2014
# novel_class_imgid_file_path = 'novel_class_imgid_of_rahman_asd_10shot_finetune_dataset_42.txt'
novel_class_imgid_file_path = 'novel_class_imgid_of_rahman_asd_10shot_finetune_dataset_42_v2.txt'
with open(novel_class_imgid_file_path) as f:
    novel_class_imgid = f.readlines()
    novel_class_imgid = [x.strip() for x in novel_class_imgid]
# target_img_dir = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/morjio_scripts/novel_class_images_ours'
target_img_dir = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/morjio_scripts/novel_class_images_ours_v2'
source_img_dir = '/data1/niehui/MSCOCO/train2014'
imgname_prefix = 'COCO_train2014_'
zero_num_max = 12
for imgid in novel_class_imgid:
    zero_num = zero_num_max - len(imgid)
    imgname = imgname_prefix + '0'*zero_num + imgid + '.jpg'
    
    src_img_path = os.path.join(source_img_dir, imgname)
    tgt_img_path = os.path.join(target_img_dir, imgname)
    shutil.copy(src_img_path, tgt_img_path)
# ipdb.set_trace()


### copy from train2017
# novel_class_imgid_file_path = 'novel_class_imgid_of_classic_fsd_finetune_dataset.txt'
# with open(novel_class_imgid_file_path) as f:
#     novel_class_imgid = f.readlines()
#     novel_class_imgid = [x.strip() for x in novel_class_imgid]
# target_img_dir = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/morjio_scripts/novel_class_images_classic'
# source_img_train17_dir = '/data1/niehui/MSCOCO/train2017'
# source_img_val17_dir = '/data1/niehui/MSCOCO/val2017'
# imgname_prefix = ''
# zero_num_max = 12
# for imgid in novel_class_imgid:
#     zero_num = zero_num_max - len(imgid)
#     imgname = imgname_prefix + '0'*zero_num + imgid + '.jpg'
#     source_img_dir = source_img_train17_dir
#     src_img_path = os.path.join(source_img_dir, imgname)
#     if not os.path.exists(src_img_path):
#         source_img_dir = source_img_val17_dir
#         src_img_path = os.path.join(source_img_dir, imgname)
#     tgt_img_path = os.path.join(target_img_dir, imgname)
    
#     shutil.copy(src_img_path, tgt_img_path)