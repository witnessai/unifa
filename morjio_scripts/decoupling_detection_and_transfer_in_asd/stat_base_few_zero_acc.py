import os 
import ipdb 


all_classes_order = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
traditional_recognition_acc = [0.97036882, 0.87288666, 0.8860781, 0.882134, 0.87635947, 0.68181818, 0.86430678, 0.85020804, 0.95137421, 0.96202532, 0.68421053, 0.82668255, 0.864, 0.89100056, 0.90632911, 0.84860789, 0.94107143, 0.98306075, 0.9872418, 0.65601023, 0.89057239, 0.61909814, 0.81268583, 0.8623348, 0.83585859, 0.8399612, 0.86116323, 0.80941176, 0.89431705, 0.82591415, 0.90875576, 0.84437221, 0.82698962, 0.79862109, 0.70800781, 0.59664804, 0.77830337, 0.90473765, 0.82972719, 0.88767551, 0.98046709, 0.90623033, 0.82098062, 0.86428982, 0.75014697, 0.8029312, 0.67053854, 0.90871369, 0.69728601, 0.84104628, 0.86819172, 0.8325821, 0.76419966, 0.91304348, 0.82370821, 0.8, 0.75660893, 0.80286169, 0.821513, 0.83580081, 0.94042891, 0.74292453, 0.80060423, 0.8820102, 0.68382353, 0.94240838, 0.87835186, 0.82608696, 0.92673993, 0.86725664, 0.71, 0.83372922, 0.73747017, 0.78290766, 0.60990712, 0.6707483, 0.88112392, 0.77192982, 0.65454545, 0.42]

BASE_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat',
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
                'vase', 'scissors', 'teddy bear', 'toothbrush')
NOVEL_CLASSES = ('airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier',)
FEW_SHOT_CLASSES_1=('train', 'parking meter', 'bear', 'snowboard', 'fork', 'hot dog', 'toaster', 'hair drier')
ZERO_SHOT_CLASSES_1=('airplane', 'cat', 'suitcase', 'frisbee', 'sandwich', 'toilet', 'mouse')


base_acc = []
novel_acc = []
few_shot_acc = []
zero_shot_acc = []
acc = traditional_recognition_acc
for i, c in enumerate(all_classes_order):
    if c in BASE_CLASSES:
        base_acc.append(acc[i])
    if c in NOVEL_CLASSES:
        novel_acc.append(acc[i])
    if c in FEW_SHOT_CLASSES_1:
        few_shot_acc.append(acc[i])
    if c in ZERO_SHOT_CLASSES_1:
        zero_shot_acc.append(acc[i])
print(sum(base_acc)/len(base_acc))
print(sum(novel_acc)/len(novel_acc))
print(sum(few_shot_acc)/len(few_shot_acc))
print(sum(zero_shot_acc)/len(zero_shot_acc))
ipdb.set_trace()