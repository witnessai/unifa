from xxlimited import new
import numpy as np 
import ipdb 




emb_path = 'data/coco/any_shot_detection/fasttext.npy'
attribute = np.load(emb_path)

new_attribute = np.zeros((81, 300))
new_attribute[:80, :] = attribute[1:, :]
new_attribute[80, :] = attribute[0, :]


save_path = 'data/coco/any_shot_detection/fasttext_switch_bg.npy'
ipdb.set_trace()
np.save(save_path, new_attribute)

ipdb.set_trace()

# word = np.loadtxt('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/pl_wordemb/word_w2v.txt', dtype='float32', delimiter=',')
# ipdb.set_trace()