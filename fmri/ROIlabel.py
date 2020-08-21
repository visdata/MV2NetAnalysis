import nibabel as nib
import numpy as np
import os 
import h5py
from scipy import io
import time
from multiprocessing.pool import Pool

FA_DK_PATH = '/home/tanzh/NM/FA_DK/'
FIBERS_PATH = '/datahouse/zhtan/NM/tract/'
OUTPUT_PATH = '/datahouse/zhtan/NM/remained_labeled_tract/'

# def jobs(f):
if __name__ == '__main__':
    f = '109_S_2200_1'
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
        #return -1
    matrix1 = np.array(mtx1['norm_fibers1'])
    matrix2 = np.array(mtx2['norm_fibers2'])
    ptnum = np.array(mtx1['ptnum'])
    matrix1 = matrix1.transpose(2,1,0)
    matrix2 = matrix2.transpose(2,1,0)
    ptnum = ptnum.transpose(1,0)
    matrix = np.concatenate((matrix1, matrix2),axis=0)
    # print('matrix: ', matrix.shape)
    fmri = np.array(nib.load(FA_DK_PATH+subject_id+'_FA_DK.nii.gz').get_data())
    # print('fmri: ', fmri.shape)
    fibers_ROI = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if matrix[j,k,2] > 49:
                matrix[j,k,2] = 49
            try:
                fibers_ROI[j, k] = fmri[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])]
            except:
                fibers_ROI[j, k] = 0
    output_filename = OUTPUT_PATH+subject_id+'_labeled_fibers.mat'
    io.savemat(output_filename, {'fiber_ROI':fibers_ROI.tolist(), 'ptnum':ptnum.tolist()}, do_compression=True)
'''
if __name__ == '__main__':
    start_time = time.time()
    # file_list = os.listdir(FA_DK_PATH)
    file_list = [x.rstrip().split('\n')[0] for x in open('subject_remained.txt', 'r')]
    
    # print(file_list)
    
    p = Pool(10)
    for f in file_list:
        p.apply_async(jobs, (f,))
    p.close()
    p.join()
    print('finish in ', time.time()-start_time)
'''
    
                    
        
        
