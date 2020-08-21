# generate data used for front-end
import numpy as np
from outliers import smirnov_grubbs as grubbs
from scipy import stats
import os
import json
import h5py
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import scikit_posthocs._outliers as so
import random
from scipy import io

warnings.filterwarnings('ignore')

JSON_PATH = '/home/tanzh/NM/streamline/features/'

# get raw data
def get_datasets(fname, path):
    
    subject_list = [x.rstrip() for x in open(fname, 'r')]
    # print(len(subject_list))
    data = []
    label = []
    info = []
    foundNum = 0
    notFoundNum = 0
    for f in subject_list:
        subject = f.split(' ')
        subject_id = subject[0]
        try:
            mtx = io.loadmat(path + subject_id + '_connectivity_end2end.mat') 
        except:
            mtx = h5py.File(path + subject_id + '_connectivity_end2end.mat', 'r')
        features = {}
        with open(JSON_PATH + subject_id[:10] + '.json', 'r') as ifile:
            features = json.load(ifile)
        
        matrix = np.array(mtx['connectivity'])
        m, n = matrix.shape
        lineArr = []
        for i in range(m):
            for j in range(i, n):
                connectivity = matrix[i][j]
                streamlineNum = features[str(i+1)+'-'+str(j+1)][0] + features[str(j+1)+'-'+str(i+1)][0]
                if int(streamlineNum) != 0:
                    length = (features[str(i+1)+'-'+str(j+1)][1] + features[str(j+1)+'-'+str(i+1)][1]) / streamlineNum
                    entropy = (features[str(i+1)+'-'+str(j+1)][2] + features[str(j+1)+'-'+str(i+1)][2]) / streamlineNum
                    curvature = (features[str(i+1)+'-'+str(j+1)][3] + features[str(j+1)+'-'+str(i+1)][3]) / streamlineNum
                    torsion = (features[str(i+1)+'-'+str(j+1)][4] + features[str(j+1)+'-'+str(i+1)][4]) / streamlineNum
                    lineArr.append([connectivity, length, entropy, curvature, torsion])
                else:
                    # 由于connectivity与其他特征分开计算，此处可能有bug，鉴于比较的时候只对单个特征比较，为保持两系统connectivity结果一致。
                    lineArr.append([connectivity, 0, 0, 0, 0])
        lineArr = list(np.array(lineArr).reshape(-1))
        data.append(lineArr)
        info.append([subject[0], subject[2], subject[3]])
        label.append(subject[1])
   
    return np.array(data), np.array(label), info

# outlier detection 
def outlierDetect(data):
    # return grubbs.test(data)
    return grubbs.max_test(data)

# distance to mean
def distanceToMean(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data*0
    else:
        return abs(data - mean) / std

# ks-test
def kstest(data):
    return stats.kstest(data, 'norm', (np.mean(data), np.std(data)))

# t-test
def ttest(data1, data2):
    scores = stats.levene(data1,data2)
    # print(scores)
    if scores.pvalue > 0.05:
        scores = stats.ttest_ind(data1,data2)
    else:
        scores = stats.ttest_ind(data1,data2,equal_var=False)
    # print(scores)
    return scores

def mannWhitneyU(biggerData, lessData):
    return stats.mannwhitneyu(biggerData, lessData, alternative='greater')

def groupComparison(data1, data2):
    signal = 0
    if len(data1) == 0 or len(data2) == 0:
        return 0.5, 0, signal
    # print(data1, data2)
    ks1 = kstest(data1)
    ks2 = kstest(data2)
    p = 0
    if ks1.pvalue>0.5 and ks2.pvalue>0.5:
        testResult = ttest(data1, data2)
        p = testResult.pvalue / 2
        signal = 1
    else:
        if  len(np.intersect1d(data1, data2))==len(data1) or len(np.intersect1d(data1, data2))==len(data2):
            return 0.5, 0, signal
        testResult = mannWhitneyU(data1, data2)
        p = testResult.pvalue
        if p > 0.5:
            testResult = mannWhitneyU(data2, data1)
            p = testResult.pvalue
        p = p / 2
        signal = 1
    return p, data1.mean() - data2.mean(), signal

# univariate feature selection
def featureSelection(data, label):
    indices = list(np.arange(data.shape[-1]))
    count = 0
    selected_dict = {}
    for index in indices:
        feature = data[:, index]
        l = label[feature!=0]
        feature = feature[feature!=0]
        if feature.size == 0:
            continue
        kstest_score = kstest(feature)
        if kstest_score.pvalue > 0.05:
            feature = outlierDetect(pd.Series(feature))
            f_index = feature.index
            feature = feature.values
            remain_l = []
            for fx in f_index:
                remain_l.append(l[fx])
            l = np.array(remain_l)
            healthy = feature[l==-1]
            ad = feature[l==1]
            if ad.size == 0 or healthy.size == 0:
                continue
            scores = ttest(healthy, ad)
            if scores.pvalue < 0.05:
                count += 1
                selected_dict[str(index)] = scores.pvalue
    return selected_dict

def getAbnormalValue(data):
    indices = list(np.arange(data.shape[-1]))
    abVal = []
    threshold = [4,0.4,0.02,0.001,0.0004]
    # indices = [0]
    for index in indices:
        feature = data[:, index]
        feature = pd.Series(feature)
        # filt data
        tsh = threshold[index%5]
        filted_feature = feature[feature>tsh]
        if len(filted_feature) == 0:
            abVal.append([0]*len(feature))
            continue
        filt = pd.Series()
        indexArr = feature.index
        f_index = filted_feature.index
        for i in indexArr:
            if i not in f_index:
                filt = filt.append(pd.Series(0, index = [i]))
        # outlier detection
        feature2 = outlierDetect(filted_feature)
        f_index = feature2.index
        indexArr = filted_feature.index
        outlier = pd.Series()
        for i in indexArr:
            if i not in f_index:
                outlier = outlier.append(pd.Series(feature.values[i], index = [i]))
        outlier = outlier.sort_values(ascending = True)
        newlist = feature2.values
        outValue = pd.Series()
        for ol_index in range(len(outlier)):
            newlist = np.append(newlist, outlier.values[ol_index])
            g = abs(outlier.values[ol_index] - newlist.mean()) / newlist.std()
            probArr, valuesArr = getGrubbsTable(newlist.size)
            gi = mapGrubbsToProb(probArr, valuesArr, g)
            # 除以几可以变动
            normVal = 1 + np.log10(gi) / 1.3
            if normVal < 0:
                normVal = 0
            outValue = outValue.append(pd.Series(normVal, index = [outlier.index[ol_index]]))
        # inlier analysis
        dvalue = distanceToMean(feature2.values)
        value = []
        for x in dvalue:
            probArr, valuesArr = getGrubbsTable(dvalue.size)
            gi = mapGrubbsToProb(probArr, valuesArr, x)
            # 除以几可以变动
            normVal = 1 + np.log10(gi) / 1.3
            if normVal < 0:
                normVal = 0
            value.append(normVal)
        value = np.array(value)
        inValue = pd.Series(value, index = feature2.index)
        # concat
        totalValue = pd.concat([filt, outValue, inValue])
        totalValue = totalValue.sort_index(ascending = True)
        abVal.append(list(totalValue.values))
    return np.array(abVal).T

def drawKStestHist(data):
    indices = list(np.arange(data.shape[-1]))
    scoresArr = []
    for index in indices:
        feature = data[:, index]
        kstest_score = kstest(feature)
        if kstest_score.pvalue <= 0.05 and kstest_score.pvalue!= 0:
            scoresArr.append(kstest_score.pvalue)
    n, bins, patches = plt.hist(x=scoresArr, bins=100, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('KStest pvalue')
    maxfreq = n.max()
    print(bins.max())
    # 设置y轴的上限
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.plot(bins[1:], n)
    plt.savefig('KStest0.png')
    plt.clf()

def KStestAnalysis(data):
    indices = list(np.arange(data.shape[-1]))
    selectedIndexArr = []
    pvalueCount = [0, 0]
    for index in indices:
        feature = data[:, index]
        kstest_score = kstest(feature)
        if kstest_score.pvalue > 0.05:
            selectedIndexArr.append(index)
            pvalueCount[1] += 1
        else:
            pvalueCount[0] += 1
    print(pvalueCount[0]/sum(pvalueCount), pvalueCount[1]/sum(pvalueCount))
    statsArr = [0,0,0,0,0]
    sampleDict = {}
    for index in selectedIndexArr:
        n = index%5
        statsArr[n] += 1
        if str(n) not in sampleDict:
            sampleDict[str(n)] = [index]
        else:
            sampleDict[str(n)].append(index)
    statsSum = sum(statsArr)
    statsArr = np.array(statsArr) / statsSum
    # print(statsSum, len(selectedIndexArr))
    print(statsArr)
    # print(len(sampleDict['0'])/statsSum,len(sampleDict['1'])/statsSum,len(sampleDict['2'])/statsSum,len(sampleDict['3'])/statsSum,len(sampleDict['4'])/statsSum)
    featureName = ['connectivity', 'length', 'entroppy', 'curvature', 'torsion']
    for ind in range(5):
        sample = sampleDict[str(ind)]
        resample = random.sample(sample, 20)
        print(resample)
        for index in resample:
            feature = data[:, index]
            n, bins, patches = plt.hist(x=feature, bins=100, color='#0504aa',
                                        alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title(featureName[ind]+'-'+str(index))
            maxfreq = n.max()
            # print(bins.max())
            # 设置y轴的上限
            plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
            plt.plot(bins[1:], n)
            plt.savefig('./kstestgreater/'+featureName[ind]+'-'+str(index)+'.png')
            plt.clf()
            
     
def getGrubbsTable(n):
    path = './'
    grubbsArr = [x.rstrip().split(' ') for x in open(path+'grubbsTable.txt', 'r')]
    probArr = grubbsArr[0]
    valuesArr = grubbsArr[n-2]
    return probArr, valuesArr

def mapGrubbsToProb(probArr, valuesArr, data):
    p = 0
    for y in range(len(valuesArr)):
        p = 1.0
        if data > float(valuesArr[y]):
            if y == 0:
                p = 0.0001
            else:
                p = float(probArr[y - 1])
            break
    return p
                
# add information
def addInfo(info, selected_dict):
    keys = list(selected_dict.keys())
    lineArr = {}
    lineArr['code'] = 0
    lineArr['data'] = [["balance","subjects","pvalue","featurename"], [10, info[0][0], selected_dict[keys[0]], keys[0]], [5, info[1][0], selected_dict[keys[1]], keys[1]]]
    print(lineArr)

# main
if __name__ == '__main__':
    data, label, info = get_datasets('subject_all_198.txt', '/datahouse/zhtan/NM/connectivity/')
    # selected_dict = featureSelection(data, label)
    # addInfo(info, selected_dict)
    print(data.shape)
    with open('all_feature_198.json', 'w') as ifile:
        json.dump(data.tolist(), ifile)
    
    
    abVal = getAbnormalValue(data)
    print(abVal.shape)
    
    with open('normalValue_198.json', 'w') as ofile:
        json.dump(abVal.tolist(), ofile)
    
    """
    normVal = np.array(abVal)
    normVal = normVal.reshape(-1)
    # normVal = normVal[normVal>=0]
    print(normVal.shape)
    n, bins, patches = plt.hist(x=normVal, bins=100, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.clf()
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('NormalValue')
    n = np.log2(n+1)
    plt.bar(x=bins[1:], height=n, width=0.05, alpha=0.8, color='yellow')
    maxfreq = n.max()
    print(maxfreq)
    # print(bins.max())
    # 设置y轴的上限
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.plot(bins[1:], n)
    plt.savefig('NormalValue.png')
    plt.clf()
    
    # group comparision
    pvalues = []
    threshold = [4,0.4,0.02,0.001,0.0004]
    count = 0
    for index in range(data.shape[-1]):
        feature = data[:,index]
        tsh = threshold[index%5]
        heal = feature[label==-1]
        ad = feature[label==1]
        heal = heal[heal>tsh]
        ad = ad[ad>tsh]
        pvalue, polar, c = groupComparison(heal, ad)
        count += c
        pvalues.append(pvalue)
    print(count/data.shape[-1], 1 - count / data.shape[-1])
    n, bins, patches = plt.hist(x=pvalues, bins=100, color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.clf()
    plt.xlabel('pValue')
    plt.ylabel('Frequency')
    plt.title('groupComparision')
    n = np.log2(n+1)
    plt.bar(x=bins[1:], height=n, width=0.005, alpha=0.8, color='yellow')
    maxfreq = n.max()
    print(maxfreq)
    # print(bins.max())
    # 设置y轴的上限
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.plot(bins[1:], n)
    plt.savefig('groupComparision.png')
    plt.clf()

    with open('all_feature.json', 'w') as ofile:
        json.dump(data.tolist(), ofile)
    """
    
    
    
    
    
   
            
            


    
    
    
    
