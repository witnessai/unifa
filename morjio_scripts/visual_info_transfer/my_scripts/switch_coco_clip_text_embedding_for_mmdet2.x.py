from xxlimited import new
import numpy as np 
import ipdb 




# emb_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg.npy'
emb_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_48_17_split.txt'
# attribute = np.load(emb_path, allow_pickle=True)
attribute = np.loadtxt(emb_path, dtype='float32', delimiter=',')
ipdb.set_trace()

new_attribute = np.zeros((attribute.shape[0], attribute.shape[1]))
new_attribute[:attribute.shape[0]-1, :] = attribute[1:, :]
new_attribute[attribute.shape[0]-1, :] = attribute[0, :]


# save_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy'
save_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2_48_17_split.npy'
# ipdb.set_trace()
np.save(save_path, new_attribute)

ipdb.set_trace()

# word = np.loadtxt('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/pl_wordemb/word_w2v.txt', dtype='float32', delimiter=',')
# ipdb.set_trace()