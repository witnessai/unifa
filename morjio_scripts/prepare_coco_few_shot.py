import argparse
import json
import os
import random
import ipdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43],
                        help="Range of seeds")
    args = parser.parse_args()
    return args


def generate_seeds(args):
    data_path = 'data/coco/annotations/instances_train2014.json'
    data = json.load(open(data_path))

    # new_all_cats 保存80个类别的详细信息
    new_all_cats = []
    for cat in data['categories']:
        new_all_cats.append(cat)

    # img_id到image完整信息的映射，例子：74478: {'license': 3, 'file_name': 'COCO_val2014_000000074478.jpg', 'coco_url': 'http://images.cocodataset.org/val2014/COCO_val2014_000000074478.jpg', 'height': 640, 'width': 428, 'date_captured': '2013-11-25 20:09:45', 'flickr_url': 'http://farm3.staticflickr.com/2753/4318988969_653bb58b41_z.jpg', 'id': 74478}
    id2img = {}
    for i in data['images']:
        id2img[i['id']] = i

    anno = {i: [] for i in ID2CLASS.keys()}
    # print(anno)的输出：{1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 13: [], 14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 27: [], 28: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: [], 37: [], 38: [], 39: [], 40: [], 41: [], 42: [], 43: [], 44: [], 46: [], 47: [], 48: [], 49: [], 50: [], 51: [], 52: [], 53: [], 54: [], 55: [], 56: [], 57: [], 58: [], 59: [], 60: [], 61: [], 62: [], 63: [], 64: [], 65: [], 67: [], 70: [], 72: [], 73: [], 74: [], 75: [], 76: [], 77: [], 78: [], 79: [], 80: [], 81: [], 82: [], 84: [], 85: [], 86: [], 87: [], 88: [], 89: [], 90: []}


    # print(a)的输出，例子：{'segmentation': [[312.29, 562.89, 402.25, 511.49, 400.96, 425.38, 398.39, 372.69, 388.11, 332.85, 318.71, 325.14, 295.58, 305.86, 269.88, 314.86, 258.31, 337.99, 217.19, 321.29, 182.49, 343.13, 141.37, 348.27, 132.37, 358.55, 159.36, 377.83, 116.95, 421.53, 167.07, 499.92, 232.61, 560.32, 300.72, 571.89]], 'area': 54652.9556, 'iscrowd': 0, 'image_id': 480023, 'bbox': [116.95, 305.86, 285.3, 266.03], 'category_id': 58, 'id': 86}
    for a in data['annotations']:
        if a['iscrowd'] == 1:
            continue
        anno[a['category_id']].append(a)
    # 所以anno是个字典，包含了每个类别所有的注释
    for i in range(args.seeds[0], args.seeds[1]):
        random.seed(i)
        # ID2CLASS.keys()是[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        for c in ID2CLASS.keys():# 遍历所有类别
            img_ids = {}
            for a in anno[c]: # 遍历第c类别的所有注释，建立起img_id到注释的映射
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            # for shots in [1, 2, 3, 5, 10, 30]:
            # for shots in [10, 30]:
            for shots in [1, 2, 3, 5]:
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    for img in imgs: # imgs is a list which contains some image ids.
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                new_data = {
                    'info': data['info'],
                    'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                save_path = get_save_path_seeds(data_path, ID2CLASS[c], shots, i)
                new_data['categories'] = new_all_cats
                with open(save_path, 'w') as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    prefix = 'full_box_{}shot_{}_trainval'.format(shots, cls)
    # save_dir = os.path.join('data', 'any_shot_ann',  'coco', 'annotations', 'seed'+str(seed))
    save_dir = os.path.join('data', 'few_shot_ann_65_15_split')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + '.json')
    return save_path


if __name__ == '__main__':
    ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}

    args = parse_args()
    generate_seeds(args)
