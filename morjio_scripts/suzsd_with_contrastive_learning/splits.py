import numpy as np

IMAGENET_ALL_CLASSES = [
    'accordion', 'airplane', 'ant', 'antelope', 'apple', 'armadillo',
    'artichoke', 'axe', 'baby bed', 'backpack', 'bagel', 'balance beam',
    'banana', 'band aid', 'banjo', 'baseball', 'basketball', 'bathing cap',
    'beaker', 'bear', 'bee', 'bell pepper', 'bench', 'bicycle', 'binder',
    'bird', 'bookshelf', 'bow tie', 'bow', 'bowl', 'brassiere', 'burrito',
    'bus', 'butterfly', 'camel', 'can opener', 'car', 'cart', 'cattle',
    'cello', 'centipede', 'chain saw', 'chair', 'chime', 'cocktail shaker',
    'coffee maker', 'computer keyboard', 'computer mouse', 'corkscrew',
    'cream', 'croquet ball', 'crutch', 'cucumber', 'cup or mug', 'diaper',
    'digital clock', 'dishwasher', 'dog', 'domestic cat', 'dragonfly',
    'drum', 'dumbbell', 'electric fan', 'elephant', 'face powder', 'fig',
    'filing cabinet', 'flower pot', 'flute', 'fox', 'french horn', 'frog',
    'frying pan', 'giant panda', 'goldfish', 'golf ball', 'golfcart',
    'guacamole', 'guitar', 'hair dryer', 'hair spray', 'hamburger',
    'hammer', 'hamster', 'harmonica', 'harp', 'hat with a wide brim',
    'head cabbage', 'helmet', 'hippopotamus', 'horizontal bar', 'horse',
    'hotdog', 'iPod', 'isopod', 'jellyfish', 'koala bear', 'ladle',
    'ladybug', 'lamp', 'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
    'lobster', 'maillot', 'maraca', 'microphone', 'microwave', 'milk can',
    'miniskirt', 'monkey', 'motorcycle', 'mushroom', 'nail', 'neck brace',
    'oboe', 'orange', 'otter', 'pencil box', 'pencil sharpener', 'perfume',
    'person', 'piano', 'pineapple', 'ping-pong ball', 'pitcher', 'pizza',
    'plastic bag', 'plate rack', 'pomegranate', 'popsicle', 'porcupine',
    'power drill', 'pretzel', 'printer', 'puck', 'punching bag', 'purse',
    'rabbit', 'racket', 'ray', 'red panda', 'refrigerator',
    'remote control', 'rubber eraser', 'rugby ball', 'ruler',
    'salt or pepper shaker', 'saxophone', 'scorpion', 'screwdriver',
    'seal', 'sheep', 'ski', 'skunk', 'snail', 'snake', 'snowmobile',
    'snowplow', 'soap dispenser', 'soccer ball', 'sofa', 'spatula',
    'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
    'strawberry', 'stretcher', 'sunglasses', 'swimming trunks', 'swine',
    'syringe', 'table', 'tape player', 'tennis ball', 'tick', 'tie',
    'tiger', 'toaster', 'traffic light', 'train', 'trombone', 'trumpet',
    'turtle', 'tv or monitor', 'unicycle', 'vacuum', 'violin',
    'volleyball', 'waffle iron', 'washer', 'water bottle', 'watercraft',
    'whale', 'wine bottle', 'zebra'
]

IMAGENET_UNSEEN_CLASSES = ['bench',
    'bow tie',
    'burrito',
    'can opener',
    'dishwasher',
    'electric fan',
    'golf ball',
    'hamster',
    'harmonica',
    'horizontal bar',
    'iPod',
    'maraca',
    'pencil box',
    'pineapple',
    'plate rack',
    'ray',
    'scorpion',
    'snail',
    'swimming trunks',
    'syringe',
    'tiger',
    'train',
    'unicycle'
]

VOC_ALL_CLASSES = np.array([
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus','cat',
    'chair', 'cow', 'diningtable', 'horse', 'motorbike','person', 
    'pottedplant', 'sheep', 'tvmonitor','car', 'dog', 'sofa', 'train'
])

COCO_ALL_CLASSES = np.array([
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
])

COCO_UNSEEN_CLASSES_65_15 = np.array([
    'airplane', 'train', 'parking meter', 'cat', 'bear', 
    'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 
    'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier'
])


COCO_ASD_FEW_SHOT_CLASSES_65_8_7 = np.array([
    'train', 'parking meter', 'bear', 'snowboard', 'fork', 'hot dog', 'toaster', 'hair drier'
])

COCO_ASD_ZERO_SHOT_CLASSES_65_8_7 = np.array([
    'airplane', 'cat', 'suitcase', 'frisbee', 'sandwich', 'toilet', 'mouse'
])

COCO_ASD_FEW_ZERO_SHOT_CLASSES_65_8_7 = np.array([
    'airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork',
               'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier'
])

VOC_UNSEEN_CLASSES = np.array(['car', 'dog', 'sofa', 'train'])

COCO_SEEN_CLASSES_48_17 = np.array([
    "toilet",
    "bicycle",
    "apple",
    "train",
    "laptop",
    "carrot",
    "motorcycle",
    "oven",
    "chair",
    "mouse",
    "boat",
    "kite",
    "sheep",
    "horse",
    "sandwich",
    "clock",
    "tv",
    "backpack",
    "toaster",
    "bowl",
    "microwave",
    "bench",
    "book",
    "orange",
    "bird",
    "pizza",
    "fork",
    "frisbee",
    "bear",
    "vase",
    "toothbrush",
    "spoon",
    "giraffe",
    "handbag",
    "broccoli",
    "refrigerator",
    "remote",
    "surfboard",
    "car",
    "bed",
    "banana",
    "donut",
    "skis",
    "person",
    "truck",
    "bottle",
    "suitcase",
    "zebra",
])

COCO_UNSEEN_CLASSES_48_17 = np.array([
    "umbrella",
    "cow",
    "cup",
    "bus",
    "keyboard",
    "skateboard",
    "dog",
    "couch",
    "tie",
    "snowboard",
    "sink",
    "elephant",
    "cake",
    "scissors",
    "airplane",
    "cat",
    "knife"
])

def get_class_labels(dataset):
    if dataset == 'coco':
        return COCO_ALL_CLASSES
    elif dataset == 'voc': 
        return VOC_ALL_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_ALL_CLASSES


def get_unseen_class_labels(dataset, split='65_15'):
    if dataset == 'coco':
        return COCO_UNSEEN_CLASSES_65_15 if split=='65_15' else COCO_UNSEEN_CLASSES_48_17
    elif dataset == 'voc': 
        return VOC_UNSEEN_CLASSES
    elif dataset == 'imagenet':
        return IMAGENET_UNSEEN_CLASSES

def get_unseen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_unseen_coco_cat_ids(split)
    elif dataset == 'voc':
        return get_unseen_voc_ids()
    elif dataset == 'imagenet':
        return get_unseen_imagenet_ids()

# asd
def get_asd_zero_shot_class_labels(dataset, split='65_8_7'):
    if dataset == 'coco':
        return COCO_ASD_ZERO_SHOT_CLASSES_65_8_7 if split=='65_8_7' else COCO_UNSEEN_CLASSES_48_17

def get_asd_zero_shot_class_ids(dataset, split='65_8_7'):
    if dataset == 'coco':
        return get_asd_zero_shot_coco_cat_ids(split)

def get_asd_zero_shot_coco_cat_ids(split='65_8_7'):
    ZERO_SHOT_CLASSES = COCO_ASD_ZERO_SHOT_CLASSES_65_8_7 if split=='65_8_7' else COCO_UNSEEN_CLASSES_48_17
    ids = np.where(np.isin(COCO_ALL_CLASSES, ZERO_SHOT_CLASSES))[0]
    return ids

def get_asd_few_zero_shot_class_labels(dataset, split='65_8_7'):
    if dataset == 'coco':
        return COCO_ASD_FEW_ZERO_SHOT_CLASSES_65_8_7 if split=='65_8_7' else COCO_UNSEEN_CLASSES_48_17

def get_asd_few_zero_shot_class_ids(dataset, split='65_8_7'):
    if dataset == 'coco':
        FEW_ZERO_SHOT_CLASSES = COCO_ASD_FEW_ZERO_SHOT_CLASSES_65_8_7 if split=='65_8_7' else COCO_UNSEEN_CLASSES_48_17
        ids = np.where(np.isin(COCO_ALL_CLASSES, FEW_ZERO_SHOT_CLASSES))[0]
        return ids

def get_asd_base_few_zero_shot_class_labels(dataset, split='65_8_7'):
    if dataset == 'coco':
        return COCO_ALL_CLASSES if split=='65_8_7' else COCO_UNSEEN_CLASSES_48_17

def get_asd_base_few_zero_shot_class_ids(dataset, split='65_8_7'):
    if dataset == 'coco':
        BASE_FEW_ZERO_SHOT_CLASSES = COCO_ALL_CLASSES if split=='65_8_7' else COCO_UNSEEN_CLASSES_48_17
        ids = np.where(np.isin(COCO_ALL_CLASSES, BASE_FEW_ZERO_SHOT_CLASSES))[0]
        return ids

def get_seen_class_ids(dataset, split='65_15'):
    if dataset == 'coco':
        return get_seen_coco_cat_ids(split)
    elif dataset == 'voc':
        return get_seen_voc_ids()
    elif dataset == 'imagenet':
        return get_seen_imagenet_ids()

def get_unseen_coco_cat_ids(split='65_15'):
    UNSEEN_CLASSES = COCO_UNSEEN_CLASSES_65_15 if split=='65_15' else COCO_UNSEEN_CLASSES_48_17
    ids = np.where(np.isin(COCO_ALL_CLASSES, UNSEEN_CLASSES))[0]
    return ids

def get_seen_coco_cat_ids(split='65_15'):
    seen_classes = np.setdiff1d(COCO_ALL_CLASSES, COCO_UNSEEN_CLASSES_65_15) if '65' in split else COCO_SEEN_CLASSES_48_17
    ids = np.where(np.isin(COCO_ALL_CLASSES, seen_classes))[0]
    return ids

def get_unseen_voc_ids():
    ids = np.where(np.isin(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES))[0]
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_seen_voc_ids():
    seen_classes = np.setdiff1d(VOC_ALL_CLASSES, VOC_UNSEEN_CLASSES)
    ids = np.where(np.isin(VOC_ALL_CLASSES, seen_classes))[0] 
    # ids = np.concatenate(([0], ids+1))
    return ids

def get_unseen_imagenet_ids():
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES))[0] + 1
    return ids

def get_seen_imagenet_ids():
    seen_classes = np.setdiff1d(IMAGENET_ALL_CLASSES, IMAGENET_UNSEEN_CLASSES)
    ids = np.where(np.isin(IMAGENET_ALL_CLASSES, seen_classes))[0]
    return ids
