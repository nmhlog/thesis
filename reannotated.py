import numpy
import glob
import os
def new_array(dim_shape,seg_id,ins_id =-100):
    arr = np.zeros((dim_shape,2))
    arr[:,0].fill(seg_id)
    arr[:,1].fill(ins_id)
    return arr  
def reannotated(data,class_idx,class_id):
    buff_data = data[class_idx]
    return np.hstack((buff_data[:,:6],new_array(buff_data[:,6:].shape[0],class_id,-100))) 

if __name__=="__main__":
    os.makedirs("data_annotated_new",exist_ok=True)
    list_txt = glob.glob("Synthetic_v3_InstanceSegmentation/**.txt")
    for idx,_ in enumerate(list_txt):
        data_txt = np.loadtxt(list_txt[idx],delimiter=",")

        ground_idx= np.where(data_txt[:,6]==0)
        building_idx= np.where(data_txt[:,6]==1)
        vegetation_idx = np.where((data_txt[:,6]>1) & (data_txt[:,6]<5)) 
        other_idx = np.where(data_txt[:,6]>4) 

        data_new = np.vstack(
            (
                reannotated(data_txt,ground_idx,0),
                data_txt[building_idx],
                reannotated(data_txt,vegetation_idx,2),
                reannotated(data_txt,other_idx,3),
            )
        ) 
        new_file_name = list_txt[idx].split("/")[-1]
        np.savetxt(f"data_annotated_new/{new_file_name}",data_new)