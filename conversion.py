# for converting text files of AAM points into numpy ndarrays in order to use them as
# a LandmarkManager object in Menpo -- to be attached to an image object

import numpy as np
from menpo.shape import PointCloud 
from menpo.landmark import LandmarkGroup, LandmarkManager

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
	print(pc)
	lmg = LandmarkGroup.init_with_all_label(pc)
	print(lmg)
	lmm = LandmarkManager()
	lmm['landmarks'] = lmg
	print(lmm)
	
	return lmm