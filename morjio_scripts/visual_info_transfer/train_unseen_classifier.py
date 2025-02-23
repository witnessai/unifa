# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from cls_models import ClsUnseenTrain
from generate import load_seen_att_with_bg
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
from splits import get_seen_class_ids
import ipdb 

# %%
# %psource ClsUnseenTrain.forward


# %%
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# %%
opt = dotdict({
    'dataset':'coco',
    'classes_split': '65_8_7',
    'class_embedding': 'data/coco/any_shot_detection/fasttext_switch_bg.npy',
    'dataroot':'data/coco/any_shot_detection/base_det',
    'trainsplit': 'train_0.6_0.3',
    
}) 
# path to save the trained classifier best checkpoint
path = 'data/coco/any_shot_detection/unseen_Classifier.pth'


# %%
seen_att, _ = load_seen_att_with_bg(opt)
unique_labels = [ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20,
       22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
       43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62,
       63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80]
classid_tolabels = {label:idx for idx, label in enumerate(unique_labels)}


# %%
unseen_classifier = ClsUnseenTrain(seen_att).cuda()

# %%
seen_features = np.load(f"{opt.dataroot}/{opt.trainsplit}_feats.npy")
seen_labels = np.load(f"{opt.dataroot}/{opt.trainsplit}_labels.npy")
print("finish loading seen features and labels")


# %%
inds = np.random.permutation(np.arange(len(seen_labels)))
total_train_examples = int (0.8 * len(seen_labels))
train_inds = inds[:total_train_examples]
test_inds = inds[total_train_examples:]
print("finish spliting train and test set")

# %%
len(test_inds)+len(train_inds), len(seen_labels)


# %%
train_feats = seen_features[train_inds]
train_labels = seen_labels[train_inds]
test_feats = seen_features[test_inds]
test_labels = seen_labels[test_inds]


# %%
# bg_inds = np.where(seen_labels==0)
# fg_inds = np.where(seen_labels>0)


# %%



# %%
class Featuresdataset(Dataset):
     
    def __init__(self, features, labels, classid_tolabels):
        self.classid_tolabels = classid_tolabels
        self.features = features
        self.labels = labels

    def __getitem__(self, idx):
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
#         import pdb; pdb.set_trace()
        
        if self.classid_tolabels is not None:
            batch_label = self.classid_tolabels[batch_label]
        return batch_feature, batch_label

    def __len__(self):
        return len(self.labels)


# %%
seen_labels.shape


# %%

dataset_train = Featuresdataset(train_feats, train_labels, classid_tolabels)
dataloader_train = DataLoader(dataset_train, batch_size=512, shuffle=True) 
dataset_test = Featuresdataset(test_feats, test_labels, classid_tolabels)
dataloader_test = DataLoader(dataset_test, batch_size=1024, shuffle=True) 
print("finish constructing dataloader")

# %%
from torch.optim.lr_scheduler import StepLR

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(unseen_classifier.parameters(), lr=1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)


# %%
min_val_loss = float("inf")


# %%



# %%



# %%
def val():
    running_loss = 0.0
    global min_val_loss
    unseen_classifier.eval()
    for i, (inputs, labels) in enumerate(dataloader_test, 0):
        inputs = inputs.cuda()
        labels = labels.cuda()
        

        outputs = unseen_classifier(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        if i % 200 == 199:
            print(f'Test Loss {epoch + 1}, [{i + 1} / {len(dataloader_test)}], {(running_loss / i) :0.4f}')
    if (running_loss / i) < min_val_loss:
        min_val_loss = running_loss / i
        state_dict = unseen_classifier.state_dict()   
        torch.save(state_dict, path)
        print(f'saved {min_val_loss :0.4f}')


# %%
for epoch in range(100):
    unseen_classifier.train()
    running_loss = 0.0
    print(epoch)
    for i, (inputs, labels) in enumerate(dataloader_train, 0):
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        optimizer.zero_grad()

        outputs = unseen_classifier(inputs)
        loss = criterion(outputs,   labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 999: 
            print(f'Train Loss {epoch + 1}, [{i + 1} / {len(dataloader_train)}], {(running_loss / i) :0.4f}')
    val()
    scheduler.step()
    ipdb.set_trace()
print('Finished Training')


# %%



