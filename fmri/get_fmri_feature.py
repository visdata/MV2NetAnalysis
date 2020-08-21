import nibabel as nib
import numpy as np
import os 
import h5py
from scipy import io
import time
from multiprocessing.pool import Pool

FMRI_PATH = '/home/tanzh/NM/FSL/'
FIBERS_PATH = '/datahouse/zhtan/NM/tract/'
LABELS_PATH = '/datahouse/zhtan/NM/remained_labeled_tract/'
OUTPUT_PATH = '/datahouse/zhtan/NM/fmri_result/'

# def jobs(f):
if __name__ == '__main__':
    f = '109_S_2200_1'
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
    filename1 = subject_id+'_fibers_FA_normed1.mat'
    filename2 = subject_id+'_fibers_FA_normed2.mat'
    mtx1 = []
    mtx2 = []
    try:
        mtx1 = h5py.File(FIBERS_PATH+filename1, 'r')
        mtx2 = h5py.File(FIBERS_PATH+filename2, 'r')
    except:
        print(subject_id+' is not dealed')
        # return -1
    matrix1 = np.array(mtx1['norm_fibers1'])
    matrix2 = np.array(mtx2['norm_fibers2'])
    ptnum = np.array(mtx1['ptnum'])
    matrix1 = matrix1.transpose(2,1,0)
    matrix2 = matrix2.transpose(2,1,0)
    ptnum = ptnum.transpose(1,0)
    matrix = np.concatenate((matrix1, matrix2),axis=0)
    # print('matrix: ', matrix.shape)
    # 读取fa，ma等数据
    fa = np.array(nib.load(FMRI_PATH+subject_id+'_EC_FA.nii.gz').get_data())
    md = np.array(nib.load(FMRI_PATH+subject_id+'_EC_MD.nii.gz').get_data())
    l1 = np.array(nib.load(FMRI_PATH+subject_id+'_EC_L1.nii.gz').get_data())
    l2 = np.array(nib.load(FMRI_PATH+subject_id+'_EC_L2.nii.gz').get_data())
    l3 = np.array(nib.load(FMRI_PATH+subject_id+'_EC_L3.nii.gz').get_data())
    # 读取label后的tract
    # labels = h5py.File(LABELS_PATH+subject_id+'_labeled_fibers.mat', 'r')
    labels = io.loadmat(LABELS_PATH+subject_id+'_labeled_fibers.mat')
    labels = np.array(labels['fiber_ROI'])
    # labels = labels.transpose(1,0)
    # print(labels.shape)
    # 计算ROI之间所有fibers上所有点fa，md等的均值
    fibers_num_all = np.zeros((70,70), dtype=np.int32)
    fa_matrix = np.zeros((70,70), dtype=np.float64)
    md_matrix = np.zeros((70,70), dtype=np.float64)
    axd_matrix = np.zeros((70,70), dtype=np.float64)
    rd_matrix = np.zeros((70,70), dtype=np.float64)
    # print(matrix[0].shape)
    for i in range(matrix.shape[0]):
        fibers = matrix[i]
        fibers_num = int(ptnum[i])
        if fibers_num < 2:
            continue
        fa_sum = 0
        md_sum = 0
        axd_sum = 0
        rd_sum = 0
        for j in range(fibers_num):
            # tract数据里的z坐标会出现超过fa的z坐标的情况
            try:
                fa_sum += fa[int(matrix[i,j,0]), int(matrix[i,j,1]), int(matrix[i,j,2])]
            except:
                # print(int(matrix[i,j,0]), int(matrix[i,j,1]), int(matrix[i,j,2]),' is out of range')
                fa_sum += 0
            try:
                md_sum += md[int(matrix[i,j,0]), int(matrix[i,j,1]), int(matrix[i,j,2])]
            except:
                md_sum += 0
            try:
                axd_sum += l1[int(matrix[i,j,0]), int(matrix[i,j,1]), int(matrix[i,j,2])]
            except:
                axd_sum += 0
            try:
                rd_sum += (l2[int(matrix[i,j,0]), int(matrix[i,j,1]), int(matrix[i,j,2])] + l3[int(matrix[i,j,0]), int(matrix[i,j,1]), int(matrix[i,j,2])])/2
            except:
                rd_sum += 0
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
        fibers_num_all[start, dest] += fibers_num
        fibers_num_all[dest, start] = fibers_num_all[start, dest]
        fa_matrix[start, dest] += fa_sum 
        fa_matrix[dest, start] = fa_matrix[start, dest] 
        md_matrix[start, dest] += md_sum 
        md_matrix[dest, start] = md_matrix[start, dest]
        axd_matrix[start, dest] += axd_sum 
        axd_matrix[dest, start] = axd_matrix[start, dest] 
        rd_matrix[start, dest] += rd_sum 
        rd_matrix[dest, start] = rd_matrix[start, dest]  
    for i in range(70):
        for j in range(70):
            if fibers_num_all[i,j] != 0:
                fa_matrix[i,j] = fa_matrix[i,j] / fibers_num_all[i,j]
                md_matrix[i,j] = md_matrix[i,j] / fibers_num_all[i,j]
                axd_matrix[i,j] = axd_matrix[i,j] / fibers_num_all[i,j]
                rd_matrix[i,j] = rd_matrix[i,j] / fibers_num_all[i,j]
    output_filename = OUTPUT_PATH + subject_id + '_fmri_result.mat'
    result = {'fa':fa_matrix.tolist(), 'md':md_matrix.tolist(), 'axd':axd_matrix.tolist(), 'rd':rd_matrix.tolist()}
    io.savemat(output_filename, result, do_compression=True)

'''
if __name__ == '__main__':
    start_time = time.time()
    # file_list = os.listdir(LABELS_PATH)
    # file_list = [x.rstrip().split('\n')[0] for x in open('subject_remained.txt')]
    file_list = ['029_S_2395_1']
    p = Pool(10)
    
    for f in file_list:
        # print(f)
        p.apply_async(jobs, (f,))
        
    p.close()
    p.join()
    print('finish in ', time.time()-start_time)
'''

