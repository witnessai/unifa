import numpy as np
import ipdb


path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_without_bg_48_17_split.txt'
# path = 'data/coco/coco_text_embedding_use_prompts_7_without_bg.txt' # 80, 512
# path = 'data/coco/coco_text_embedding_use_prompts_7_originalorder_without_bg.txt'
# path = 'data/coco/word_w2v_with_learnable_bg_65_15_ori.txt' # shape is 300, 81

text_embedding = np.loadtxt(path, dtype='float32', delimiter=',')


text_embedding = text_embedding.T # shape is 512, 80/65
combine = np.zeros((text_embedding.shape[0], text_embedding.shape[1]+1))
combine[:, 1:] = text_embedding
combine[:, 0] = np.mean(text_embedding, 1)
combine = combine.T # shape is  (81/66, 512)
# ipdb.set_trace()
np.savetxt('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_48_17_split.txt', combine, delimiter=',')

# text_embedding = text_embedding.T # shape is 512, 80
# combine = np.zeros((text_embedding.shape[0], text_embedding.shape[1]+1))
# combine[:, :-1] = text_embedding
# combine[:, -1] = np.mean(text_embedding, 1)
# combine = combine.T # shape is  (81, 512)combine, delimiter=',')
# np.savetxt('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2_48_17_split.txt', combine)