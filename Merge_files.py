import numpy as np
import glob
import pandas as pd
import json
import os
import torch
import tqdm


coordShiftPath = r'stpls3d/val/coordShift.json'
coordShift = json.load(open(coordShiftPath))
coordDir = r'val/coords_offsets'
semanticDir = r'val/semantic'
insMaskDir = r'val/predicted_masks'
print_each_block = True
mergerDir = r"merge_files"
os.makedirs(mergerDir,exist_ok=True)
coordFilePaths = glob.glob(coordDir + '/*.npy')
pointcloud_id= [5,10,15,20,25]

for k in range(len(pointcloud_id)):
    buff = [i for i in coordFilePaths if i.split("/")[-1].startswith(pointcloud_id[k])]
    outPath = f'merge_{pointcloud_id[k]}.txt'
    outFile = open(outPath,'w')
    insLabel = 1
    for coordFilePath in buff:
        fileName = os.path.basename(coordFilePath).strip('.npy')
        print (fileName)
        xyz = np.load(coordFilePath)
        semantic = np.load(os.path.join(semanticDir,fileName+'.npy'))
        insMaskFilePathList = sorted(glob.glob(insMaskDir + '/%s*.txt' %fileName))

        ins = np.zeros(len(xyz))
        for insMaskPath in insMaskFilePathList:
            insMask = pd.read_csv(insMaskPath, delimiter=',', header=None).values
            insMask = np.squeeze(insMask, axis=1)
            ins[insMask==1] = insLabel
            insLabel+=1
        if print_each_block :
            torch.save((xyz[:,:3],semantic,ins), f"{mergerDir}/"+"merge "+coordFilePath.split("/")[-1][:-4]+".pth")
        # break
        xyz[:,:3] += np.array([float(value) for value in coordShift[fileName]])
        xyz[:,:3] += np.array([float(value) for value in coordShift['globalShift']])
        for i in range(len(xyz)):
            outFile.write("%f,%f,%f,%d,%d\n" %(xyz[i][0],xyz[i][1],xyz[i][2],semantic[i],ins[i]))

    outFile.close()
