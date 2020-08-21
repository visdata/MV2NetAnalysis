import json
import os

path = '/home/tanzh/NM/streamline/JSON4/'
outputPath = '/home/tanzh/NM/streamline/features/'
for f in os.listdir(path):
    features = {}
    with open(path + f, 'r') as ifile:
        data = json.load(ifile)
        for ROI in data['ROIs']:
            ROIStartId = ROI['ROIStartId']
            ROIEndId = ROI['ROIEndId']
            if int(ROIStartId) == 0 or int(ROIEndId) == 0:
                continue
            streamlineArray = ROI['streamlineArray']
            streamlineNum = ROI['StreamlineNum']
            length = 0.0
            entropy = 0.0
            curvature = 0.0
            torsion = 0.0
            for s in streamlineArray:
                length += s['Attribute']['length']
                entropy += s['Attribute']['entropy']
                curvature += s['Attribute']['curvature']
                torsion += s['Attribute']['torsion']
            line = [streamlineNum, length, entropy, curvature, torsion]
            features[str(ROIStartId) + '-' + str(ROIEndId)] = line
    with open(outputPath + f, 'w') as ofile:
        json.dump(features, ofile)
    
