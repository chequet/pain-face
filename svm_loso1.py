import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from AAM import AAMfitter, prune_labels, generate_normcapps
from setup import splitData
from sklearn.svm import LinearSVC
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score
import numpy as np
import menpo.io as mio
from menpofit.aam import HolisticAAM


def LOSO(testsubj, destination, trstart=0, trfinish=None, testleaveout=False):
	# separate training and test sets
	tr, trl, tst, tstl, aam = splitData('/Data/Images/', testsubj)
	# if no end point specified, continue to end of dataset
	if trfinish is None:
		trfinish = len(tr)
	# train aam on aam dataset
	fitter = AAMfitter(aam,HolisticAAM)
	# generate SVM inputs
	traincapps, trainerrors = generate_normcapps(fitter,tr,trstart,trfinish,destination + 'train/')
	if testleaveout == False:
		testcapps, testerrors = generate_normcapps(fitter,tst,0,len(tst), destination + 'test/')
	mio.export_pickle(trl, destination + 'train/labels/trl')
	mio.export_pickle(tstl, destination + 'test/labels/tstl')

def classify(path):
	# load data -- throw error for empty import path
	print('loading data...')
	traincapps = mio.import_pickles(path+'train/')
	trainerrors = mio.import_pickles(path+'train/errs/')
	testcapps = mio.import_pickles(path+'test/')
	# it's ok if there are no failed CAPPs in the test set as it is short
	# (but there are certainly failed CAPPs in the training set)
	testerrors = []
	try:
		testerrors = mio.import_pickles(path+'test/errs/')
	except ValueError:
		pass
	trainlabels = mio.import_pickles(path+'train/labels/')[0]
	testlabels = mio.import_pickles(path+'test/labels/')[0]
	# check nothing has gone hopelessly wrong
	assert len(trainlabels) == len(traincapps) + len(trainerrors), "Inconsistent training set/labels size"
	assert len(testlabels) == len(testcapps) + len(testerrors), "Inconsistent test set/labels size"

	# remove labels for images which did not produce CAPPs
	trainlabels = prune_labels(trainlabels,trainerrors)
	testlabels = prune_labels(testlabels,testerrors)
	# again check nothing gone very wrong
	assert len(trainlabels) == len(traincapps)
	assert len(testlabels) == len(testcapps)

	# sometimes CAPPs are different lengths which messes up PCA
	# so we must pad short CAPPs with zeros
	lens = [len(t) for t in traincapps]
	# if all lens are the same, proceed, otherwise 
	if not all(x == lens[0] for x in lens):
		print('identified inconsistent CAPP lengths, padding...')
		# identify max len, identify locations of shorter CAPPs
		maxlen = np.max(lens)
		pads = [i for i in range(len(lens)) if lens[i] != maxlen]
		# check if all short CAPPs are the same length
		slens = [lens[i] for i in range(len(lens)) if i in pads]
		assert all(x==slens[0] for x in slens), "extra inconsistent lengths, need more help!"
		diff = maxlen - slens[0]
		# create vector of zeros to append
		pad = np.zeros(diff)
		# append padding vector to any short CAPPS
		tc = [np.append(traincapps[i],pad) if i in pads else traincapps[i] for i in range(len(traincapps))]
		# sanity check
		assert len(traincapps) == len(tc), "we lost some CAPPs somewhere"
		assert all(len(x)==len(tc[0]) for x in tc), "padding did not work"
	# otherwise, assume everything worked and proceed... 	

	# fit PCA model incrementally to training data
	print('fitting PCA model...')
	pca = IncrementalPCA(n_components=30, batch_size=40)
	tc = pca.fit_transform(traincapps)
	# generate PCA version of test data 
	print('transforming test set...')
	tsc = pca.transform(testcapps)
	# classify
	print('fitting classifier...')
	clf = LinearSVC()
	clf.fit(tc,trainlabels)
	print('making prediction...')
	pred = clf.predict(tsc)
	# check accuracy 
	print(accuracy_score(testlabels,pred))



