import numpy as np
import ipdb 

# 合并softlabels和sampgtlabels两个npy文件
# root = 'data/coco/any_shot_detection/base_few_shot_det/multi_labels/zero_shot'
# process_traindata = False
# if process_traindata:
#     trainsplit = 'train_0.6_0.3'
#     softlabels = np.load(f"{root}/{trainsplit}_softlabels.npy", allow_pickle=True)
#     sampgtlabels = np.load(f"{root}/{trainsplit}_sampgtlabels.npy", allow_pickle=True)
# else: 
#     testsplit = 'test_0.6_0.3'
#     softlabels = np.load(f"{root}/{testsplit}_softlabels.npy", allow_pickle=True)
#     sampgtlabels = np.load(f"{root}/{testsplit}_sampgtlabels.npy", allow_pickle=True)

# new_softlabels = np.zeros((10271105, 80))
# new_softlabels_pos = 0
# for softlab, sampgtlab in zip(softlabels, sampgtlabels):
#     uniquelab = np.unique(sampgtlab)
#     for row_soft, row_gt in zip(softlab, sampgtlab):
#         uniquelab_recorder = {x:0 for x in uniquelab}
#         for i in range(len(row_gt)):
#             gt = row_gt[i]
#             uniquelab_recorder[gt] = max(uniquelab_recorder[gt], row_soft[i])
#         class_loc = list(uniquelab_recorder.keys())
#         class_iou = list(uniquelab_recorder.values())
#         new_softlabels[new_softlabels_pos][class_loc] = class_iou 
#         new_softlabels_pos += 1
#         if new_softlabels_pos % 100000 == 0:
#             print(int(new_softlabels_pos/100000))

# if process_traindata:
#     np.save(f'{root}/{trainsplit}_newsoftlabels.npy', new_softlabels)
# else: 
#     np.save(f'{root}/{testsplit}_newsoftlabels.npy', new_softlabels)


# 加载合并后的文件进行查看
root = 'data/coco/any_shot_detection/base_few_shot_det/multi_labels/zero_shot'
testsplit = 'test_0.6_0.3'
data = np.load(f"{root}/{trainsplit}_newsoftlabels.npy", allow_pickle=True)