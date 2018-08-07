import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from AAM import AAMfitter
from setup import splitData
from sklearn.svm import LinearSVC
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score
import numpy as np
import menpo.io as mio
from menpofit.aam import HolisticAAM


def LOSO(testsubj, destination, trstart=0, trfinish=None):
	# separate training and test sets
	tr, trl, tst, tstl, aam = splitData('/Data/Images/', testsubj)
	if trfinish is None:
		trfinish = len(tr)
	# train aam on aam dataset
	fitter = AAMfitter(aam,HolisticAAM)
	# generate SVM inputs
	traincapps, trainerrors = generate_normcapps(fitter,tr,trstart,trfinish,destination + 'train/')
	testcapps, testerrors = generate_normcapps(fitter,tst,0,len(tst), destination + 'test/')

def classify():