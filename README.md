# 脑网络数据处理说明文档

## 简称说明：
T1影像：T1加权成像
DWI影像：弥散加权成像
FA：Fractional anisotropy
MD：Mean diffusivity
AxD：Axial diffusivity
RD：Radial diffusivity
ROI：Region of Interest

## 一、脑影像数据：
数据集来自ADNI（Alzheimer’s Disease Neuroimaging Initiative），包含202人的核磁共振影像数据，其中包括50个健康个体，72个轻度认知损伤个体，38个重度认知损伤个体和42个阿尔兹海默症患者，年龄区间为55到90，共有男性120名和女性82名。核磁共振影像数据包括T1加权成像（T1 weighted image），和弥散加权成像（Diffusion weighted image）。

## 二、脑影像预处理和纤维示踪：
DWI影像通过FSL（FMRIB Software Library）的BET（Brain Extraction Tool）工具进行去骨头处理。去骨头之后的DWI影像使用FSL进行涡流校正（Eddy-current correction），这一步的目的是消除EPI（Echo‐planar Imaging）获取影像过程中的失真问题。然后，通过FSL的DTIFIT（Diffusion Tensors Image Fitting）工具，由弥散张量模型得到各向异性（FA）、平均弥散系数（MD）、轴向弥散系数（AxD）和径向弥散系数（RD）。神经纤维的示踪使用的是Camino（http://cmic.cs.ucl.ac.uk/camino/)，一个开源的DWI影像处理工具。纤维示踪使用的是PICo（Probabilistic Index of Connectivity method）算法，一个基于概率的示踪算法。具体方法为把种子设置在FA值大于0.3的体素上，从种子开始使用蒙特卡洛算法沿着概率密度图模拟流线的生成，概率密度图由步长为1mm的4阶龙格-库塔（Runge–Kutta）算法插值得到的局部最大值进行估计。神经纤维的最大弯曲角度设置为45度/体素，流线的生成在FA值小于0.2的体素上停止。

## 三、脑影像配准和纤维标签：
在进行配准之前，需要对T1影像进行去骨头处理，去骨后的T1影像使用FreeSurfer，一个开源的脑影像处理工具，进行模板配准。FreeSurfer使用的是Desikan-Killiany模板，包含70个ROI。T1影像模板划分结束之后，使用ANTs与预处理生成的FA影像进行配准，模板T1作为输入图像，FA影像作为参考影像，配准后得到FA的模板划分。根据预处理得到的神经纤维与FA影像的对应关系，可以从FA的模板划分的到神经纤维所经过的ROI，通过神经纤维两端所处的ROI组成的二元组，可以把神经纤维分成不同的纤维束。

## 四、特征值生成：
用于对脑网络进行分析的特征包括神经纤维的强度（Strength），各向异性（FA）、平均弥散系数（MD）、轴向弥散系数（AxD）、径向弥散系数（RD）。神经纤维的强度表示的是ROI之间相连的纤维的数目，各向异性、平均弥散系数、轴向弥散系数和径向弥散系数表示的是连接不同ROI的神经纤维束中纤维上所有体素对应的值取平均。

## 五、处理流程：
```
（1）	使用BrainSuite对T1图像进行去骨处理，得到T1_Stripped；
（2）	使用eddy对DWI数据进行eddy correct；
（3）	使用BET对DWI数据进行处理得到mask；
（4）	使用DTIFIT对DWI，bvec，bval，mask处理得到FA，V1，V2，V3
（5）	使用FreeSurfer处理T1_Stripped得到DK模板下的T1_DK；
（6）	使用ANTs对FA和T1_DK进行配准得到FA_DK；
（7）	根据FA_DK对Fiber进行label得到DK模板下的labeled_fiber；
（8）	对labeled_fiber进行计算得到特征值。
```
