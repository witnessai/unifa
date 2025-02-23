import matplotlib.pyplot as plt
import seaborn as sns
import ipdb
import numpy as np



classnames = ['airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier', 'background']

gen_classifier_acc = [0.7916, 0.9005, 0.2872, 0.8990, 0.3687, 0.5263, 0.4200, 0.7827, 0.8128, 0.8989, 0.1147, 0.7659, 0.5588, 0.7009, 0.4242, 0.8722]
det_classifier_acc_wrong = [0.00915994, 0.00019516, 0.08712785, 0.00113397, 0.01249099, 0., 0.00232751, 0., 0.00105611, 0., 0.00114112, 0.00082516, 0.42949485, 0., 0.01895735, 0.07382922]
det_classifier_acc = [0.5588767, 0.634758, 0.27189142, 0.40207355, 0.80975258, 0.20657486, 0.37090607, 0.47719131, 0.19656766, 0.21297108, 0.23678205, 0.39690107, 0.33133828, 0.18803419, 0.08056872, 0.99712085]
residual_between_gen_and_det = list(np.array(gen_classifier_acc) - np.array(det_classifier_acc)) # 生成分类器和检测分类器之差
det_ap = [0.235, 0.187, 0.105, 0.238, 0.369, 0.066, 0.235, 0.121, 0.026, 0.079, 0.059, 0.192, 0.210, 0.038, 0.003, 0]

# ax = sns.barplot(classnames, gen_classifier_acc)
# ax.set_ylim(0, 1) # 限制y的值为[0, 1]
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
# plt.show()

# ax = sns.barplot(classnames, det_classifier_acc)
# ax.set_ylim(0, 1) # 限制y的值为[0, 1]
# ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
# plt.show()


ax = sns.barplot(classnames, residual_between_gen_and_det)
ax.set_ylim(-0.5, 0.7) 
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.show()

ipdb.set_trace()