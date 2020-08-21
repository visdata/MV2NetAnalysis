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

FMRI_RESULT_PATH = '/datahouse/zhtan/NM/fmri_result/'
CONNECT_PATH = '/datahouse/zhtan/NM/connectivity/'

# get raw data
def get_datasets(fname):
    # file_list = os.listdir(FMRI_RESULT_PATH)
    subject_list = [x.rstrip() for x in open(fname, 'r')]
    # print(subject_list)
    data = []
    label = []
    info = []
    
    for f in subject_list:
        subject = f.split(' ')
        subject_id = subject[0]
        
        #try:
            # mtx = h5py.File(path + f, 'r')
        try:
            mtx = io.loadmat(CONNECT_PATH + subject_id + '_connectivity_end2end.mat') 
        except:
            mtx = h5py.File(CONNECT_PATH + subject_id + '_connectivity_end2end.mat', 'r')
        features = io.loadmat(FMRI_RESULT_PATH + subject_id + '_fmri_result.mat')
        
        
        fa = np.array(features['fa'])
        md = np.array(features['md'])
        axd = np.array(features['axd'])
        rd = np.array(features['rd'])
        matrix = np.array(mtx['connectivity'])
        
        m, n = matrix.shape
        lineArr = []
        for i in range(m):
            for j in range(i, n):
                lineArr.append([fa[i][j], md[i][j], axd[i][j], rd[i][j], matrix[i][j]])
                
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



def getAbnormalValue(data):
    indices = list(np.arange(data.shape[-1]))
    abVal = []
    threshold = [0.05, 0.0005, 0.0005, 0.0005, 4]
    for index in indices:
        feature = data[:, index]
        feature = pd.Series(feature)
        # filt data
        tsh = threshold[index%5]
        filted_feature = feature[feature>tsh]
        # filted_feature = feature
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

def draw(data, title):
    n, bins, patches = plt.hist(x=data, bins=100, 
                                alpha=0.7, rwidth=0.85)
    # plt.clf()
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    # n = np.log10(n+1)
    # plt.bar(x=bins[1:], height=n, width=0.05, alpha=0.8, color='yellow')
    maxfreq = n.max()
    # print(maxfreq)
    # print(bins.max())
    # 设置y轴的上限
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.plot(bins[1:], n)
    plt.savefig(title+'.png')
    plt.clf()


# main
if __name__ == '__main__':
    
    data, label, info = get_datasets('subject_all_198.txt')
    print(data.shape)
    
    with open('all_feature_fmri_198.json', 'w') as ifile:
        json.dump(data.tolist(), ifile)
    '''
    data = []
    with open('all_feature_fmri.json', 'r') as ifile:
        data = json.load(ifile)
    data = np.array(data)
    '''
    abVal = getAbnormalValue(data)
    print(abVal.shape)
    with open('normalValue_fmri_198.json', 'w') as ofile:
        json.dump(abVal.tolist(), ofile)
    
    '''
    feature_name = ['FA', 'MD', 'AxD', 'RD', 'Strength']
    indices = np.arange(12425)
    for ind in range(len(feature_name)):
        title = feature_name[ind]
        indices_mask = [True if x%5 == ind else False for x in indices]
        indices_mask = np.array(indices_mask)
        data_chip = data[:, indices_mask]
        data_chip = data_chip.reshape(-1)
        data_chip= data_chip[data_chip!=0]
        draw(data_chip, title)
    indices_mask = [True if x%5 == ind else False for x in indices]
    indices_mask = np.array(indices_mask)
    data = data[:,indices_mask]
    data = data.reshape(-1)
    print(data[data!=0].shape[0]/data.shape[0])
    '''
    
    
    
    
    
    
    
    
   
            
            


    
    
    
    
