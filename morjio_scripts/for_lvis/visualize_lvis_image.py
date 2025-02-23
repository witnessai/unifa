import os
import cv2
from lvis import LVIS, LVISVis
import ipdb 

# 初始化 LVIS 数据集
data_dir = '/data1/niehui/'
lvis = LVIS(os.path.join(data_dir, 'lvis/annotations/lvis_v1_val.json'))

# 初始化 LVISVis 实例
vis = LVISVis(lvis_gt=lvis, img_dir=os.path.join(data_dir, 'MSCOCO/val2017'))



# 可视化图像
vis.vis_img(
    img_id=397133,  # 图像 ID
    show_boxes=True,  # 可视化边界框
    show_segms=True,  # 可视化分割掩模
    show_classes=True,  # 可视化类别名称
    # cat_ids_to_show=[1, 2, 3],  # 可视化指定的类别
)

ipdb.set_trace()