import nibabel as nib
import numpy as np
import os 
import h5py
from scipy import io
import time
from multiprocessing.pool import Pool

FMRI_PATH = '/home/tanzh/NM/FSL/'
FIBERS_PATH = '/datahouse/zhtan/NM/tract/'
PATH = '/datahouse/zhtan/NM/fmri_result/'
OUTPUT_PATH = '/datahouse/zhtan/NM/fmri_fibers/'
# OUTPUT_PATH = '/datahouse/zhtan/NM/fmri_fibers_axd/'

def jobs(f):
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

    fibers_fa = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    fibers_md = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    fibers_rd = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    fibers_axd = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if matrix[j,k,2] > 49:
                matrix[j,k,2] = 49
            try:
                fibers_fa[j, k] = fa[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])]
                fibers_md[j, k] = md[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])]
                fibers_rd[j, k] = (l2[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])] + l3[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])])/2
                fibers_axd[j, k] = l1[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])]
            except:
                fibers_fa[j, k] = 0
                fibers_md[j, k] = 0
                fibers_rd[j, k] = 0
                fibers_axd[j, k] = 0

    output_filename_fa = OUTPUT_PATH + subject_id + '_fibers_fa.mat'
    output_filename_md = OUTPUT_PATH + subject_id + '_fibers_md.mat'
    output_filename_rd = OUTPUT_PATH + subject_id + '_fibers_rd.mat'
    output_filename_axd = OUTPUT_PATH + subject_id + '_fibers_axd.mat'
    io.savemat(output_filename_fa, {'fiber_fa':fibers_fa.tolist(), 'ptnum':ptnum.tolist()}, do_compression=True)
    io.savemat(output_filename_md, {'fiber_fa':fibers_md.tolist(), 'ptnum':ptnum.tolist()}, do_compression=True)
    io.savemat(output_filename_rd, {'fiber_fa':fibers_rd.tolist(), 'ptnum':ptnum.tolist()}, do_compression=True)
    io.savemat(output_filename_axd, {'fiber_fa':fibers_axd.tolist(), 'ptnum':ptnum.tolist()}, do_compression=True)

def jobs_axd(f):
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
    l1 = np.array(nib.load(FMRI_PATH+subject_id+'_EC_L1.nii.gz').get_data())

    fibers_fa = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    fibers_axd = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=np.float64)
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            if matrix[j,k,2] > 49:
                matrix[j,k,2] = 49
            try:
                fibers_fa[j, k] = fa[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])]
                fibers_axd[j, k] = l1[int(matrix[j,k,0]), int(matrix[j,k,1]), int(matrix[j,k,2])]
            except:
                fibers_fa[j, k] = 0
                fibers_axd[j, k] = 0

    output_filename_fa = OUTPUT_PATH + subject_id + '_fibers_fa.mat'
    output_filename_axd = OUTPUT_PATH + subject_id + '_fibers_axd.mat'
    io.savemat(output_filename_fa, {'fiber_fa':fibers_fa.tolist(), 'ptnum':ptnum.tolist()}, do_compression=True)
    io.savemat(output_filename_axd, {'fiber_fa':fibers_axd.tolist(), 'ptnum':ptnum.tolist()}, do_compression=True)


if __name__ == '__main__':
    '''
    start_time = time.time()
    # file_list = os.listdir(PATH)
    file_list = [x.rstrip().split(' ')[0] for x in open('subject_all_198.txt')]
    # file_list = ['098_S_4050_1','127_S_4148_1','027_S_2245_1','109_S_2200_1']
    p = Pool(10)
    
    for f in file_list:
        # print(f)
        p.apply_async(jobs, (f,))
        
    p.close()
    p.join()
    print('finish in ', time.time()-start_time)
    '''
    img = nib.load('/home/tanzh/NM/FSL_TO_DK/'+'003_S_4081_1_FA_TO_DK.nii.gz')
    fa = np.array(nib.load('/home/tanzh/NM/FSL_TO_DK/'+'003_S_4081_1_FA_TO_DK.nii.gz').get_data())
    print(img.affine)
    

