import numpy as np
import ipdb 

################################check labels begin
## test feature labels
# path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/test_0.6_0.3_labels.npy'
## base training feature labels
# path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_labels_base_training.npy'
## fine tuning feature labels
# path3 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_labels_fine_tuning.npy'

# labels = np.load(path1)
# print(set(labels))
# ipdb.set_trace()
################################check labels end




################################merge labels begin for 3shot
## merge base traininig and fine tuning labels
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1)
# label1[label1==15] == 20 # remap bg label to unify base training and fine tuning labels
# label2 = np.load(label_path2)
# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
################################merge labels end for 3shot



################################merge labels begin for 5shot
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1)
# label1[label1==15] == 20 # remap bg label to unify base training and fine tuning labels
# label2 = np.load(label_path2)
# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
################################merge labels end for 5shot


################################merge labels begin for 3shot, fsce
## merge base traininig and fine tuning labels
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1)
# label1[label1==15] == 20 # remap bg label to unify base training and fine tuning labels
# label2 = np.load(label_path2)
# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
################################merge labels end for 3shot, fsce



################################merge labels begin for 5shot, fsce
## merge base traininig and fine tuning labels
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1)
# label1[label1==15] == 20 # remap bg label to unify base training and fine tuning labels
# label2 = np.load(label_path2)
# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
################################merge labels end for 5shot, fsce



################################merge labels begin for 3shot, fsce add cpe
## merge base traininig and fine tuning labels
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1)
# label1[label1==15] == 20 # remap bg label to unify base training and fine tuning labels
# label2 = np.load(label_path2)
# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
################################merge labels end for 3shot, fsce add cpe



################################merge labels begin for 5shot, fsce add cpe
## merge base traininig and fine tuning labels
label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot/train_0.6_0.3_labels_base_training.npy'
label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot/train_0.6_0.3_labels_fine_tuning.npy'
label1 = np.load(label_path1)
label1[label1==15] == 20 # remap bg label to unify base training and fine tuning labels
label2 = np.load(label_path2)
feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot/train_0.6_0.3_feats_base_training.npy'
feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot/train_0.6_0.3_feats_fine_tuning.npy'
feat1 = np.load(feat_path1)
feat2 = np.load(feat_path2)
label = np.concatenate((label1, label2), axis=0)
feat = np.concatenate((feat1, feat2), axis=0)
save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot/train_0.6_0.3_labels.npy'
save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot/train_0.6_0.3_feats.npy'
ipdb.set_trace()
np.save(save_label_path, label)
np.save(save_feat_path, feat)
################################merge labels end for 5shot, fsce add cpe
