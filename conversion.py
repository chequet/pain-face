# for converting text files of AAM points into numpy ndarrays in order to use them as
# a LandmarkManager object in Menpo -- to be attached to an image object
# also for converting text files of PSPI labels into a dictionary

import numpy as np
from menpo.shape import PointCloud 
from menpo.landmark import LandmarkManager
from os import walk

def landmarkConverter(filename):
	file = open(filename, 'r')
	lms = file.read()
	lmsplit = lms.splitlines()
	coords = []
	for item in lmsplit:
		item = item.split()
		point = [float(item[1]),float(item[0])]
		coords.append(point)
	file.close()
	ra = np.array(coords)
	pc = PointCloud(ra)
	lmm = LandmarkManager()
	lmm['landmarks'] = pc
	
	return lmm

def labelConverter(directory):
	trainingpaths = []
	for root, dirs, files in walk('/Data/Frame_Labels/PSPI/'):
	    for file in files:
	        trainingpaths.append(str(root) +'/'+ str(file))
	        
	labels = {}
	for path in trainingpaths:
	    file = open(path, 'r')
	    val = float(file.read())
	    if val != 0:
	        label = 1
	    else:
	        label = 0
	    key = path.replace('/Data/Frame_Labels/PSPI/','')
	    labels[key] = label
	return labels
    