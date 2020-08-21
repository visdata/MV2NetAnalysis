import nibabel as nib
import numpy as np
import os 
import h5py
from scipy import io
import time
from multiprocessing.pool import Pool


LABELS_PATH = '/datahouse/zhtan/NM/remained_labeled_tract/'
OUTPUT_PATH = '/datahouse/zhtan/NM/connectivity/'

def jobs(f):
    # 读取roi映射关系
    # idx = h5py.File('/datahouse/zhtan/NM/index.mat', 'r')
    idx = io.loadmat('/datahouse/zhtan/NM/index.mat')
    idx = np.array(idx['index'])
    idx = idx.reshape(-1)
    # print(idx.shape)
    # 读取tract数据
    subject_id = f.split('_')
    subject_id = '_'.join(subject_id[:4])
    print('Processing '+subject_id)
    
    # 读取label后的tract
    # labels = h5py.File(LABELS_PATH+subject_id+'_labeled_fibers.mat', 'r')
    labels = []
    try:
        labels = io.loadmat(LABELS_PATH+subject_id+'_labeled_fibers.mat')
    except:
        return -1
    labels = np.array(labels['fiber_ROI'])
    # labels = labels.transpose(1,0)
    # print(labels.shape)
    
    connect_matrix = np.zeros((70,70), dtype=np.float64)
    
    # print(matrix[0].shape)
    for i in range(labels.shape[0]):
        label = labels[i]
        for l in range(len(label)):
            if label[l] == 0:
                continue
            else:
                if int(label[l])-1 > 2034:
                    label[l] = 0
                    print(int(label[l])-1, 'out of range')
                else:
                    label[l] = idx[int(label[l])-1]
        label = label[label!=0]
        if len(label) < 2:
            continue
        start = int(label[0]) - 1
        dest = int(label[-1]) - 1
        connect_matrix[start, dest] += 1
        connect_matrix[dest, start] = connect_matrix[start, dest] 
        
    output_filename = OUTPUT_PATH + subject_id + '_connectivity_end2end.mat'
    result = {'connectivity':connect_matrix.tolist()}
    io.savemat(output_filename, result, do_compression=True)


if __name__ == '__main__':
    start_time = time.time()
    # file_list = os.listdir(LABELS_PATH)
    # file_list = [(x.rstrip().split('\t'))[0] for x in open('subject_remained.txt', 'r')]
    file_list = ['109_S_2200_1']
    # jobs(file_list[0]) 
    p = Pool(10)
    for f in file_list:
        p.apply_async(jobs, (f,))   
    p.close()
    p.join()
    print('finish in ', time.time()-start_time)
    


