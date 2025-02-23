import ipdb 
import numpy as np 

class_embedding = 'data/coco/any_shot_detection/fasttext_switch_bg.npy'
data = np.load(class_embedding) # data.shape is (81, 300)
sim_matrix = np.zeros((81, 81))

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')



for i in range(81):
    for j in range(81):
        if i != j:
            sim_matrix[i, j] = np.dot(data[i], data[j]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[j]))
sim_matrix = np.around(sim_matrix, 2)
print(sim_matrix)

# novel classes cat id
COCO_NOVEL_CLASSES =  [5, 17, 33, 34, 54, 70, 74]
# all classes cat id 
COCO_ALL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
COCO_IDMAP = {cat_id: idx for idx, cat_id in enumerate(COCO_ALL_CLASSES)}

for i, c in enumerate(COCO_NOVEL_CLASSES):
    max_value = np.max(sim_matrix[COCO_IDMAP[c]])
    location = np.where(np.isin(sim_matrix[COCO_IDMAP[c]], max_value))[0][0]
    print(COCO_IDMAP[c], location)
    # print(CLASSES[COCO_IDMAP[c]], ' is most similar to ', CLASSES[location] )

ipdb.set_trace()

# airplane  is most similar to  car
# cat  is most similar to  dog
# suitcase  is most similar to  backpack
# frisbee  is most similar to  dog
# sandwich  is most similar to  pizza
# toilet  is most similar to  toothbrush
# mouse  is most similar to  keyboard

# 4 2
# 15 16
# 28 24
# 29 16
# 48 53
# 61 79
# 64 66

