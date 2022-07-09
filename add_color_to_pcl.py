import numpy as np
import glob
import pandas as pd
import json
import os
import torch
import tqdm

CLASS_COLOR = {
    -100:[0,0,0],
    0: [0, 0, 0],
    1: [143, 223, 142],
    2: [171, 198, 230],
    3: [0, 120, 177],
    4: [255, 188, 126],
    5: [189, 189, 57],
    6: [144, 86, 76],
    7: [255, 152, 153],
    8: [222, 40, 47],
    9: [197, 176, 212],
    10: [150, 103, 185],
    11: [200, 156, 149],
    12: [0, 190, 206],
    13: [252, 183, 210],
    14: [219, 219, 146]
}
COLOR25=[[231, 54, 54],
[231, 98, 54],
[231, 142, 54],
[231, 187, 54],
[231, 231, 54],
[187, 231, 54],
[142, 231, 54],
[98, 231, 54],
[54, 231, 54],
[54, 231, 98],
[54, 231, 142],
[54, 231, 187],
[54, 231, 231],
[54, 187, 231],
[54, 142, 231],
[54, 98, 231],
[54, 54, 231],
[98, 54, 231],
[142, 54, 231],
[187, 54, 231],
[231, 54, 231],
[231, 54, 187],
[231, 54, 142],
[231, 54, 98],
[231, 54, 54]]
COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])


mapping_COLOR_SEG= lambda x : CLASS_COLOR[x]
mapping_COLOR_INS= lambda x : COLOR25[x]

def create_nx3_colormatrix(array_exp):
    return np.zeros((len(array_exp),3))

def coloring_ins_not_black(ins_id,ins_idx_color,dict_map):
    ins_color = create_nx3_colormatrix(ins_id)
    ins_colored = create_nx3_colormatrix(ins_id[ins_idx_color])
    for i in range(len(ins_colored)):
        ins_colored[i] = dict_map[int(ins_id[ins_idx_color][i])]
    ins_color[ins_idx_color] = ins_colored
    return ins_color

def get_color_for_instance_and_semantic(sem_id,ins_id):
    sem_color = create_nx3_colormatrix(sem_id)
    for idx,val in enumerate(sem_id):
        sem_color[idx] = CLASS_COLOR[val]
    inst_color_not_black = np.where((sem_id!=0) & (sem_id!=-100))[0]   
    dict_map = {}
    for idx,val in enumerate(np.unique(ins_id[inst_color_not_black])):
        if val == 0 or val == -100:
            dict_map.update({val:np.array([0,0,0])})
        dict_map.update({val.astype(int):mapping_COLOR_INS(idx%25)})
    ins_color = coloring_ins_not_black(ins_id,inst_color_not_black,dict_map)
    return sem_color,ins_color

if __name__ =="__main__":
    """
    Filtering only certain values
    """
    outFile = open("list_folder.txt",'w')
    for f_name in glob.glob("stpls3d/val/**.pth"):
        _,_,seg_id,_ = torch.load(f_name)
        for ids in np.unique(seg_id).astype(int):
            if ids in [9, 10,11,12]:
                outFile.write(str(np.unique(seg_id).astype(int))+","+f_name+"\n")
                print(str(np.unique(seg_id).astype(int))+","+f_name)
                break
    outFile.close()
    """
    Read Filter data
    """
    with open("list_folder.txt","r") as file_name:
        filter_dataList = file_name.readlines()
    data_list = [i.strip().split(",")[-1].split("/")[-1] for i in filter_dataList]  
    """
    Visualization for each block
    """
    os.makedirs("save",exist_ok=True)


    for i,value in tqdm.tqdm(enumerate(data_list),total=len(data_list)) :
        merge= torch.load(f"merge_files/merge {data_list[i]}")
        coor_merge = merge[0]
        sort_merge= np.lexsort((coor_merge[:,0], coor_merge[:,1],coor_merge[:,2]))
        coor_merge,sem_id_merge,ins_id_merge = merge[0][sort_merge],merge[1][sort_merge],merge[2][sort_merge]
        sem_merge_color, ins_merge_color= get_color_for_instance_and_semantic(sem_id_merge,ins_id_merge)
        np.savetxt(f"save/merge {data_list[i][:-4]}.txt",np.concatenate((coor_merge,sem_merge_color,ins_merge_color),axis=1),fmt="%.6f %.6f %.6f %d %d %d %d %d %d")
        val = torch.load(f"stpls3d/val/{data_list[i]}")
        coor_val = val[0]
        sort_val= np.lexsort((coor_val[:,0], coor_val[:,1],coor_val[:,2]))
        coor_val,sem_id_val,ins_id_val = val[0][sort_val],val[2][sort_val],val[3][sort_val]
        sem_val_color, ins_val_color= get_color_for_instance_and_semantic(sem_id_val,ins_id_val)

        np.savetxt(f"save/val {data_list[i][:-4]}.txt",np.concatenate((coor_val,sem_val_color,ins_val_color),axis=1),fmt="%.6f %.6f %.6f %d %d %d %d %d %d")