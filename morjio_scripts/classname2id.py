import ipdb 
import numpy as np 


ID2CLASS = { 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush", }

CLASS2ID = dict()
for key in ID2CLASS:
    value = ID2CLASS[key]
    CLASS2ID[value] = key

fs_one = ['train', 'parking meter', 'bear', 'snowboard', 'fork', 'hot dog', 'toaster', 'hair drier']
fs_two = ['train', 'parking meter', 'cat', 'bear', 'snowboard', 'fork', 'toilet', 'toaster']
fs_three = ['train', 'parking meter', 'cat', 'bear', 'snowboard', 'fork', 'toilet', 'toaster']
fs_four = ['train', 'parking meter', 'bear', 'snowboard', 'fork',  'mouse', 'toaster', 'hair drier']
zs_four = ['airplane',  'cat', 'suitcase', 'frisbee',  'sandwich', 'hot dog', 'toilet']

name_lists = [zs_four]
catid_lists = []
for name_list in name_lists:
    tmp = []
    for name in name_list:
        tmp.append(CLASS2ID[name])
    catid_lists.append(tmp)
print(catid_lists)
# ipdb.set_trace()





COCO_ALL_CLASSES = np.array(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'])

BASE_CLASSES = np.array(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat',
                'traffic light', 'fire hydrant',
                'stop sign', 'bench', 'bird', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie',
                'skis', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'orange', 'broccoli', 'carrot',
                'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'tv', 'laptop',
                'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'toothbrush'])
FEW_SHOT_CLASSES_set4 = np.array(['train', 'parking meter', 'bear', 'snowboard', 'fork',  'mouse', 'toaster', 'hair drier'])
ZERO_SHOT_CLASSES_set4 = np.array(['airplane',  'cat', 'suitcase', 'frisbee',  'sandwich', 'hot dog', 'toilet'])

# label id
print(np.where(np.isin(COCO_ALL_CLASSES, ZERO_SHOT_CLASSES_set4)))
ipdb.set_trace()