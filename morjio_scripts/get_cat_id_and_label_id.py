import numpy as np 

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

ID2CLASS = { 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush"}

CLASS2ID = {'person': 1, 'bicycle': 2, 'car': 3, 'motorcycle': 4, 'airplane': 5, 'bus': 6, 'train': 7, 'truck': 8, 'boat': 9, 'traffic light': 10, 'fire hydrant': 11, 'stop sign': 13, 'parking meter': 14, 'bench': 15, 'bird': 16, 'cat': 17, 'dog': 18, 'horse': 19, 'sheep': 20, 'cow': 21, 'elephant': 22, 'bear': 23, 'zebra': 24, 'giraffe': 25, 'backpack': 27, 'umbrella': 28, 'handbag': 31, 'tie': 32, 'suitcase': 33, 'frisbee': 34, 'skis': 35, 'snowboard': 36, 'sports ball': 37, 'kite': 38, 'baseball bat': 39, 'baseball glove': 40, 'skateboard': 41, 'surfboard': 42, 'tennis racket': 43, 'bottle': 44, 'wine glass': 46, 'cup': 47, 'fork': 48, 'knife': 49, 'spoon': 50, 'bowl': 51, 'banana': 52, 'apple': 53, 'sandwich': 54, 'orange': 55, 'broccoli': 56, 'carrot': 57, 'hot dog': 58, 'pizza': 59, 'donut': 60, 'cake': 61, 'chair': 62, 'couch': 63, 'potted plant': 64, 'bed': 65, 'dining table': 67, 'toilet': 70, 'tv': 72, 'laptop': 73, 'mouse': 74, 'remote': 75, 'keyboard': 76, 'cell phone': 77, 'microwave': 78, 'oven': 79, 'toaster': 80, 'sink': 81, 'refrigerator': 82, 'book': 84, 'clock': 85, 'vase': 86, 'scissors': 87, 'teddy bear': 88, 'hair drier': 89, 'toothbrush': 90}

few_shot_name_set_one = ['train', 'parking meter', 'bear', 'snowboard', 'fork', 'hot dog', 'toaster', 'hair drier']
few_shot_name_set_two = ['train', 'parking meter', 'cat', 'bear', 'snowboard', 'fork', 'toilet', 'toaster']
zero_shot_name_set_one = ['airplane', 'cat', 'suitcase', 'frisbee', 'sandwich', 'toilet', 'mouse']
zero_shot_name_set_two = ['airplane',  'suitcase', 'frisbee', 'sandwich', 'hot dog', 'mouse', 'hair drier']

few_shot_name_set_one_cat_id = []
for name in few_shot_name_set_one:
    few_shot_name_set_one_cat_id.append(CLASS2ID[name])
few_shot_name_set_one_label_id = np.where(np.isin(CLASSES, few_shot_name_set_one))[0]
print(few_shot_name_set_one_cat_id, few_shot_name_set_one_label_id)

few_shot_name_set_two_cat_id = []
for name in few_shot_name_set_two:
    few_shot_name_set_two_cat_id.append(CLASS2ID[name])
few_shot_name_set_two_label_id = np.where(np.isin(CLASSES, few_shot_name_set_two))[0]
print(few_shot_name_set_two_cat_id, few_shot_name_set_two_label_id)

zero_shot_name_set_one_cat_id = []
for name in zero_shot_name_set_one:
    zero_shot_name_set_one_cat_id.append(CLASS2ID[name])
zero_shot_name_set_one_label_id = np.where(np.isin(CLASSES, zero_shot_name_set_one))[0]
print(zero_shot_name_set_one_cat_id, zero_shot_name_set_one_label_id)

zero_shot_name_set_two_cat_id = []
for name in zero_shot_name_set_two:
    zero_shot_name_set_two_cat_id.append(CLASS2ID[name])
zero_shot_name_set_two_label_id = np.where(np.isin(CLASSES, zero_shot_name_set_two))[0]
print(zero_shot_name_set_two_cat_id, zero_shot_name_set_two_label_id)

