import numpy as np
import torch
from numpy import linalg as LA

from splits import get_unseen_class_ids ,get_seen_class_ids, get_asd_zero_shot_class_ids, get_asd_few_zero_shot_class_ids

import ipdb 


def load_all_att(opt):
    attribute = np.load(opt.class_embedding) # class_embedding='MSCOCO/fasttext.npy'
    labels = np.arange(len(attribute))
    attribute/=LA.norm(attribute, ord=2) 
    return torch.from_numpy(attribute), torch.from_numpy(labels)

def load_seen_att_with_bg(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = np.concatenate((get_seen_class_ids(opt.dataset, split=opt.classes_split), [80]))
    return attribute[classes_ids], labels[classes_ids]

def load_unseen_att_with_bg(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = np.concatenate((get_unseen_class_ids(opt.dataset, split=opt.classes_split), [80]))
    return attribute[classes_ids], labels[classes_ids]

def load_unseen_att(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = get_unseen_class_ids(opt.dataset, split=opt.classes_split) # opt.dataset='coco', opt.classes_split='65_15'
    return attribute[classes_ids], labels[classes_ids]

def load_zero_shot_att(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = get_asd_zero_shot_class_ids(opt.dataset, split=opt.classes_split) # opt.dataset='coco', opt.classes_split='65_15'
    return attribute[classes_ids], labels[classes_ids]

def load_few_zero_shot_att(opt):
    attribute, labels = load_all_att(opt)
    classes_ids = get_asd_few_zero_shot_class_ids(opt.dataset, split=opt.classes_split) # opt.dataset='coco', opt.classes_split='65_15'
    return attribute[classes_ids], labels[classes_ids]
