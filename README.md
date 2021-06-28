# MV2Net系统数据分析文档
More details on the system backend can be found in:
https://github.com/visdata/MV2NetVis

## 数据集
|数据集|存储位置|文件格式|概要说明|
|-----|-------|--------|-------|
|DWI|/home/tanzh/NM/DWI|nii文件，bvals和bvecs文件|DWI文件|
|FA|/home/tanzh/NM/FA|nii文件|FA文件|
|T1|/home/tanzh/NM/T1|nii.gz文件|去骨后的T1|
|FA_DK|/home/tanzh/NM/FA_DK|nii.gz文件|使用ANTs对FA和T1_DK配准后的文件|
|T1_DK|* |*	|使用FreeSurfer对T1进行模板配准得到的文件|
|Fiber|/datahouse/zhtan/NM/tract|mat文件|神经纤维文件|
|Labeled fiber|/datahouse/zhtan/NM/remained_labeled_tract|mat文件|根据模板对点进行映射后的神经纤维文件|
|Fiber strength|/datahouse/zhtan/NM/connectivity|mat文件|神经纤维强度|
|Geometric feature|/home/tanzh/NM/streamline/features|json文件|几何特征数据|
|Diffusion feature|/datahouse/zhtan/NM/fmri_result|mat文件|扩散特征数据|

### DWI
#### 获取方式
金焰老师提供的数据。

#### 详细描述

#### 使用方法
参考链接：https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FDT/UserGuide
（1）生成mask
工具：BET
命令行代码：
bet <input> <output> [options]

（2）DTIFIT
工具：DTIFIT
命令行代码：
dtifit  -k,--data       dti data file
        -o,--out        Output basename
        -m,--mask       Bet binary mask file
        -r,--bvecs      b vectors file
        -b,--bvals      b values file









