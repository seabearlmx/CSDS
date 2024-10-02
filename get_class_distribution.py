import numpy as np
import os.path as osp
import os
from PIL import Image
import glob
from collections import Counter
import xlwt
import pandas as pd


src_id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

trg_id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
id_mapping = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle"]

# src_save_path = r'F:/AdaptationSemanicSeg/data/GTA5/labels/train/folder'
src_save_path = r'/mnt/datasets/Cityscapes/gtFine/train/'

nums = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0}

cate=[src_save_path+x for x in os.listdir(src_save_path) if os.path.isdir(src_save_path+x)]
print(cate)
for idx,folder in enumerate(cate):
    print(folder)
    for im in glob.glob(folder+'/*_labelTrainIds.png'):
        print(im)
        src_label = Image.open(im)
        src_label = src_label.resize((512, 1024), Image.NEAREST)
        src_label = np.asarray(src_label, np.float32)
        src_label_copy = src_label
        # src_label_copy = 255 * np.ones(src_label.shape, dtype=np.float32)
        # for k, v in trg_id_to_trainid.items():
        #     src_label_copy[src_label == k] = v
        counter = Counter(src_label_copy.flatten())
        # print(counter)
        keys = counter.keys()
        # print(keys)
        for i in range(19):
            if i in keys:
                nums[i] += counter[i]
                # print(counter[i])
#
lists = []
for keys, value in nums.items():
    # 键和值都要
    temp = [keys,value]
    lists.append(temp)
print(lists)
#
#
# columns = ["category_id", "numbers"]
# dt = pd.DataFrame(lists, columns=columns)
# dt.to_excel("cityscapes.xlsx", index=0)


