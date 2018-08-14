import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from AAM import AAMfitter, generate_normcapps
from setup import splitData, prune_labels, check_proportions
from sklearn.svm import LinearSVC
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score, classification_report, average_precision_score
import numpy as np
import menpo.io as mio
from menpofit.aam import HolisticAAM
from menpo.base import LazyList
import time


def LOSO(destination, loadfitter=False, testsubj=None, trstart=0, trfinish=None, testleaveout=False, loaddata= False):
	if loaddata:
		# load data from pickles
		print("...loading aam data...")
		aam = mio.import_pickle(destination+'aam/aamimgs.pkl')
		print("...loading training images...")
		tr = mio.import_pickles(destination+'train/imgs/',verbose=True)
		tr = [item for sublist in tr for item in sublist]
	else:
		# separate training and test sets
		tr, trl, tst, tstl, aam = splitData('/Data/Images/', destination, testsubj)
	if loadfitter:
		print("loading fitter...")
		fitter = mio.import_pickle(destination+'aam/fitter.pkl')
	else:
		# train aam on aam dataset
		fitter = AAMfitter(aam,HolisticAAM)
		mio.export_pickle(fitter,destination+'aam/fitter.pkl')
	# if no end point specified, continue to end of dataset
	if trfinish is None:
		trfinish = len(tr)	
	# generate SVM inputs in segments
	generate_normcapps(fitter,tr,trstart,trfinish,destination+'train/')
	if loaddata:
		print("...loading test images...")
		tst = mio.import_pickle(destination+'tst/imgs/tstimgs.pkl')
	if testleaveout == False:
		generate_normcapps(fitter,tst,0,len(tst), destination + 'test/')


def classify(path, already_pruned=False):
	# load data -- throw error for empty import path
	print('loading data...')
	traincapps = mio.import_pickles(path+'train/', verbose=True)
	trainerrors = mio.import_pickles(path+'train/errs/', verbose=True)
	testcapps = mio.import_pickles(path+'test/', verbose=True)
	# it's ok if there are no failed CAPPs in the test set as it is short
	# (but there are certainly failed CAPPs in the training set)
	testerrors = []
	try:
		testerrors = mio.import_pickles(path+'test/errs/',verbose=True)
	except ValueError:
		pass
	trainlabels = mio.import_pickle(path+'train/labels/trl.pkl',verbose=True)
	testlabels = mio.import_pickle(path+'test/labels/tstl.pkl',verbose=True)
	# check nothing has gone hopelessly wrong
	assert len(trainlabels) == len(traincapps) + len(trainerrors), "Inconsistent training set/labels size"
	assert len(testlabels) == len(testcapps) + len(testerrors), "Inconsistent test set/labels size"
	print("training pain proportion {}".format(check_proportions(trainlabels)))
	print("test pain proportion {}".format(check_proportions(testlabels)))
	if already_pruned:
		trainlabels = mio.import_pickle(path + 'train/labels/trl_pruned.pkl',verbose=True)
		testlabels = mio.import_pickle(path + 'test/labels/tstl_pruned.pkl',verbose=True)
	else:
		# remove labels for images which did not produce CAPPs
		print('pruning labels...')
		trainlabels = prune_labels(trainlabels,trainerrors)
		testlabels = prune_labels(testlabels,testerrors)
		# again check nothing gone very wrong
		assert len(trainlabels) == len(traincapps)
		assert len(testlabels) == len(testcapps)
		mio.export_pickle(trainlabels,path+'train/labels/trl_pruned.pkl',overwrite=True)
		mio.export_pickle(testlabels,path+'test/labels/tstl_pruned.pkl', overwrite=True)

	# sometimes CAPPs are different lengths which messes up PCA
	# so we must pad short CAPPs with zeros
	lens = [len(t) for t in traincapps]
	# identify max length of CAPP
	maxlen = np.max(lens)
	# if all lens are the same, proceed, otherwise 
	if not all(x == lens[0] for x in lens):
		tc = np.ndarray([len(traincapps),maxlen],dtype=float)
		print('identified inconsistent CAPP lengths...')
		# divide traincapps into batches
		batch = range(0, len(traincapps), 150)
		temp = np.ndarray([150,maxlen], dtype = float)
		for i in batch:
			print('checking training CAPPs {}-{}/{}'.format(str(i),str(i+150),str(len(traincapps))))
			temp = traincapps[i:i+150]
			templens = [len(t) for t in temp]
			pads = [i for i in range(len(templens)) if templens[i] != maxlen]
			print('padding {} CAPPs'.format(str(len(pads))))
			# check all short capps in this batch are the same length
			slens = [templens[i] for i in range(len(templens)) if i in pads]
			assert all(x==slens[0] for x in slens), "extra inconsistent lengths, need more help!"
			diff = maxlen - slens[0]
			# create vector of zeros to append
			pad = np.zeros(diff)
			temp = [np.append(temp[i],pad) if i in pads else temp[i] for i in range(len(temp))]
			tc[i:i+150] = temp
		# sanity check
		#assert not traincapps.has_nanvalues(), "Found NaN values"
		assert len(traincapps) == len(tc), "we lost some CAPPs somewhere"
		assert all(len(x)==len(tc[0]) for x in tc), "padding did not work"
		#return traincapps, tc
		print(tc.shape)
		#traincapps = LazyList(tc)
	# otherwise, assume everything worked and proceed... 	
	# fit PCA model incrementally to training data
	print('fitting PCA model...')
	pca = IncrementalPCA(n_components=30, batch_size=30)
	tca = pca.fit_transform(traincapps)
	# generate PCA version of test data 
	print('transforming test set...')
	tsc = pca.transform(testcapps)
	# classify
	print('fitting classifier...')
	clf = LinearSVC()
	clf.fit(tca,trainlabels)
	mio.export_pickle(clf,path+'clf/fittedclf.pkl')
	print('making prediction...')
	pred = clf.predict(tsc)
	mio.export_pickle(pred,path+'clf/pred.pkl')
	# check accuracy 
	print(accuracy_score(testlabels,pred))
	class_names = ['no pain', 'pain']
	print(classification_report(testlabels,pred, target_names=class_names))
	


