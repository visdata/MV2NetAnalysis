# MV2Net系统数据分析文档
More details on the system backend can be found in:
https://github.com/visdata/MV2NetVis

## 1.数据集
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

### 1.1DWI
#### 获取方式
金焰老师提供的数据。

#### 1.2详细描述
DWI（Diffusion-weighted imaging）文件。

#### 1.3使用方法
```
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
```
### 2.FA
#### 2.1获取方式：
DWI作为输入，由DTIFIT生成。

#### 2.2详细描述：
FA（部分各向异性）文件。

#### 2.3使用方法：
使用ANTs对FA和T1_DK配准，得到FA_DK。


### 3.T1
#### 3.1获取方式：
服务器上已有。

#### 3.2详细描述：
去骨后的T1脑影像。

#### 3.3使用方法：
作为FreeSurfer的输入，与DK模板进行匹配，生成T1_DK文件。


### 4.FA_DK
#### 4.1获取方式：
由ANTs对FA和T1_DK进行配准得到。

#### 4.2详细描述：
与DK模板配准后的FA（部分各向异性）文件。

#### 4.3使用方法：
作为ROIlabel.py的输入，对神经纤维进行模板划分。


### 5.T1_DK
#### 5.1获取方式：
由FreeSurfer对T1进行模板配准得到。

#### 5.2详细描述：
与DK模板配准后的去骨T1脑影像。

#### 5.3使用方法：
用于ANTs配准。


### 6.Fiber
#### 6.1获取方式：
服务器上已有。

#### 6.2详细描述：
神经纤维文件。
储存着神经纤维上点的坐标。

#### 6.3使用方法：
作为ROIlabel.py的输入，输出模板划分后的神经纤维。


### 7.Labeled_iber
#### 7.1获取方式：
由ROIlabel.py生成。

#### 7.2详细描述：
模板划分后的神经纤维文件。
神经纤维上的点映射到模板ROI。

#### 7.3使用方法：
作为ROIlabel.py的输入，输出模板划分后的神经纤维。


### 8.Fiber strength
#### 8.1获取方式：
由generate_connectivity.py生成。

#### 8.2详细描述：
神经纤维的强度。

#### 8.3使用方法：
作为getOriginData.py和getOriginDataForFMri.py的输入，生成脑网络的神经纤维强度特征数据。

### 9.Geometric feature
#### 9.1获取方式：
基于中山大学提供的几何特征数据（JSON格式），由streamlineProcess.py生成。

#### 9.2详细描述：
脑网络的几何特征。

#### 9.3使用方法：
作为getOriginData.py的输入，生成用于MV2Net系统的几何特征数据。

### 10.Diffusion feature
#### 10.1获取方式：
基于DTIFIT的结果，由get_fmri_feature.py生成。

#### 10.2详细描述：
脑网络的扩散特征。

#### 10.3使用方法：
作为getOriginDataForFMri.py的输入，生成用于MV2Net系统的扩散张量特征数据。






