import torch 
from torchvision import models 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.init as init
import torch.nn as nn
import torch.optim as optim
import cv2
from PIL import Image
import ipdb 
import os
import numpy as np
import json
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
np.set_printoptions(suppress=True)

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
all_classes_disorder = BASE_CLASSES+NOVEL_CLASSES

def weights_init(m):
    classname = m.__class__.__name__ 
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)

def cal_acc(fsl_coco2014val):
    base_ap = []
    novel_ap = []
    few_shot_ap = []
    zero_shot_ap = []
    ap = fsl_coco2014val
    for i, c in enumerate(all_classes_disorder):
        if c in BASE_CLASSES:
            base_ap.append(ap[i])
        if c in NOVEL_CLASSES:
            novel_ap.append(ap[i])
        if c in FEW_SHOT_CLASSES_1:
            few_shot_ap.append(ap[i])
        if c in ZERO_SHOT_CLASSES_1:
            zero_shot_ap.append(ap[i])
    print(sum(base_ap)/len(base_ap))
    print(sum(novel_ap)/len(novel_ap))
    print(sum(few_shot_ap)/len(few_shot_ap))
    print(sum(zero_shot_ap)/len(zero_shot_ap))

class COCO_bbox_crop(Dataset):
    def __init__(self, data_root='data/coco', img_dir='bbox_crop_train2014'):
        self.data_root = data_root
        self.img_dir = img_dir
        self.img_root_path = os.path.join(data_root, img_dir)
        self.img_names = os.listdir(self.img_root_path)
        self.img_names.sort()
        self.name2label = dict()
        
        print("%s dataset has %d images!" % (img_dir, len(self.img_names)))
    
    def transform(self, image):
        imageTransform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = imageTransform(image)
        return image

    def __getitem__(self, idx):
        while True:
            img_name = self.img_names[idx]
            img_path = os.path.join(self.data_root, os.path.join(self.img_dir, img_name))
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            # ipdb.set_trace()
            if min(w, h) > 5:
                break
            else:
                idx = np.random.randint(0, len(self.img_names))
        
        img = self.transform(img)
        label = np.array(int(img_name.split('.')[0].split('_')[-1])-1)
        label = torch.from_numpy(label)
        self.name2label[img_name] = label

        # ipdb.set_trace()
        # pre-process
        return img, label, img_name
    
    def __len__(self):
        return len(self.img_names)


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # 取掉model的后1层
        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(2048, 80) #加上一层参数修改好的全连接层
 
    def forward(self, x):
        x = self.resnet_layer(x)
        # ipdb.set_trace()
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x
 

batch_size = 64
data_root = '/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_fsl/65_15_split'
trainset = COCO_bbox_crop(data_root, img_dir='base_train')
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
testset = COCO_bbox_crop(data_root, img_dir='gfsl_test')
testsetloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)

resnet = models.resnet101(pretrained=True)
resnet = Net(resnet)
resnet = resnet.cuda()

optimizer_resnet = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss().cuda()
class_num = 80
epoch_max = 0
print('Start BaseTraining!')

labels_counter_basetrain = np.zeros(80)

resnet.train()
for epoch in range(epoch_max):
    print("Epoch %d:" % epoch )
    for i, data in enumerate(trainsetloader, 0):
        optimizer_resnet.zero_grad()
        inputs, labels, _ = data
        # for cat_id in range(0, 80):
        #     labels_counter_basetrain[cat_id] += len(labels[labels==cat_id])
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = resnet(inputs)
        # labels = labels.cpu()
        # one_hot_labels = torch.zeros(labels.shape[0], class_num).scatter_(1, labels.view(-1, 1), 1).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_resnet.step()
        if i % 200 == 0:
            print('Epoch: %d, iter: %d, loss: %f' % (epoch, i, loss.item()))
            # break
print('Finish BaseTraining!')

model_save_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/decoupling_detection_and_transfer_in_asd/fsl_finetuning_v1.pth'
# if not os.path.exists(model_save_path):
#     torch.save({
#         'epoch': epoch_max,
#         'model': resnet.state_dict(),
#         'optimizer': optimizer_resnet.state_dict(),
#     }, model_save_path)


# with open('co_ocurrence_matrix_of_COCO_v3cnn.txt') as fd:
#     co_occurrence_matrix = np.loadtxt(fd, delimiter=',')
# with open('imgname2label_of_bboxcropval2014.json') as fd:
#     imgname2label = json.load(fd)
resnet_model_params = torch.load(model_save_path)
resnet.load_state_dict(resnet_model_params['model'])
resnet.eval()
acc_tp_all = 0
acc_total_all = 0
acc_tp_per_cls = np.zeros(class_num)
acc_total_per_cls = np.zeros(class_num)
confusion_matrix = np.zeros((class_num, class_num))
# unreasonable_pred_total_per_cls = np.zeros((10, class_num)) # [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5], 7个阈值

labels_counter_eval = np.zeros(80)
for i, data in enumerate(tqdm(testsetloader)):
    # print(i)
    # if i > 10: break
    inputs, labels, img_names = data  # getitem有做减一的操作
    # for cat_id in range(0, 80):
    #     labels_counter_eval[cat_id] += len(labels[labels==cat_id])
    inputs, labels = inputs.cuda(), labels.cuda()
    outputs = resnet(inputs)
    values, inds = torch.max(outputs, 1)
    acc_tp_all += torch.sum(inds == labels) # 预测标签等于gt的为tp
    acc_total_all += labels.shape[0] # 类别总数为gt的长度
    for cls_id in range(class_num):
        acc_total_per_cls[cls_id] += torch.sum(labels == cls_id) #每个类别id的gt长度
        labels_id = labels[labels == cls_id] # gt label为id的全部挑选
        inds_id = inds[labels == cls_id] # gt label为id的所有位置的预测标签都挑选
        acc_tp_per_cls[cls_id] += torch.sum(inds_id == labels_id)
        
    # confusion_matrix
    for cls_id in range(class_num):                                                  
        inds_id = inds[labels == cls_id]
        for pred_id in range(class_num):
            confusion_matrix[cls_id, pred_id] += torch.sum(inds_id == pred_id)


print(confusion_matrix)
print(1.0*acc_tp_all/acc_total_all)
print(np.mean(acc_tp_per_cls))
print(np.mean(acc_total_per_cls))
print('Finish fsl_basetrain model eval!')
fsl_coco2014val = acc_tp_per_cls/(acc_total_per_cls+0.000001)
cal_acc(fsl_coco2014val)
ipdb.set_trace()
# np.savetxt('unreasonable_pred_of_coco_bbox_vanillaCNN_0527morethres.txt', unreasonable_pred_total_per_cls, delimiter=',', fmt='%d')
# np.savetxt('confusion_matrix_of_coco_bbox_vanillaCNN_0527morethres.txt', confusion_matrix, delimiter=',', fmt='%d')


trainset = COCO_bbox_crop(data_root, img_dir='finetune_subset')
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16) 

resnet.train()
# 冻结除fc以外的所有层
for module in resnet.modules():
    if module._get_name() != 'Linear':
        print('layer: ',module._get_name())
        for param in module.parameters():
            param.requires_grad_(False)
    elif module._get_name() == 'Linear':
        for param in module.parameters():
            param.requires_grad_(True)
optimizer_resnet = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)

criterion = nn.CrossEntropyLoss().cuda()
class_num = 80
epoch_max = 12
print('Start finetuning!')

labels_counter_finetuning = np.zeros(80)

for epoch in range(epoch_max):
    print("Epoch %d:" % epoch )
    for i, data in enumerate(trainsetloader, 0):
        optimizer_resnet.zero_grad()
        inputs, labels, _ = data
        # for cat_id in range(0, 80):
        #     labels_counter_finetuning[cat_id] += len(labels[labels==cat_id])
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = resnet(inputs)
        # labels = labels.cpu()
        # one_hot_labels = torch.zeros(labels.shape[0], class_num).scatter_(1, labels.view(-1, 1), 1).cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_resnet.step()
        if i % 200 == 0:
            print('Epoch: %d, iter: %d, loss: %f' % (epoch, i, loss.item()))
            # break
    ipdb.set_trace()
print('Finish finetuning!')


resnet.eval()
acc_tp_all = 0
acc_total_all = 0
acc_tp_per_cls = np.zeros(class_num)
acc_total_per_cls = np.zeros(class_num)
confusion_matrix = np.zeros((class_num, class_num))
# unreasonable_pred_total_per_cls = np.zeros((10, class_num)) # [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5], 7个阈值
# ipdb.set_trace()
for i, data in enumerate(testsetloader, 0):
    # print(i)
    # if i > 10: break
    inputs, labels, img_names = data  # getitem有做减一的操作
    inputs, labels = inputs.cuda(), labels.cuda()
    outputs = resnet(inputs)
    values, inds = torch.max(outputs, 1)
    acc_tp_all += torch.sum(inds == labels) # 预测标签等于gt的为tp
    acc_total_all += labels.shape[0] # 类别总数为gt的长度
    for cls_id in range(class_num):
        acc_total_per_cls[cls_id] += torch.sum(labels == cls_id) #每个类别id的gt长度
        labels_id = labels[labels == cls_id] # gt label为id的全部挑选
        inds_id = inds[labels == cls_id] # gt label为id的所有位置的预测标签都挑选
        acc_tp_per_cls[cls_id] += torch.sum(inds_id == labels_id)
        
    # confusion_matrix
    for cls_id in range(class_num):                                                  
        inds_id = inds[labels == cls_id]
        for pred_id in range(class_num):
            confusion_matrix[cls_id, pred_id] += torch.sum(inds_id == pred_id)
    # ipdb.set_trace()

print(confusion_matrix)
print(1.0*acc_tp_all/acc_total_all)
print(np.mean(acc_tp_per_cls))
print(np.mean(acc_total_per_cls))
print(np.mean(acc_tp_per_cls/acc_total_per_cls))
print('Finish fsl_finetuning model eval!')

fsl_coco2014val = acc_tp_per_cls/(acc_total_per_cls+0.000001)
cal_acc(fsl_coco2014val)

model_save_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/decoupling_detection_and_transfer_in_asd/fsl_finetuning.pth'
if not os.path.exists(model_save_path):
    torch.save({
        'epoch': epoch_max,
        'model': resnet.state_dict(),
        'optimizer': optimizer_resnet.state_dict(),
    }, model_save_path)
ipdb.set_trace()