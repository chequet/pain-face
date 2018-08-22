import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import menpo.io as mio
from conversion import landmarkConverter
from os import walk
import numpy as np
import joblib

def process(image, crop_proportion=0.2, max_diagonal=400):
	# converts images to B&W, adds landmarks, crops to landmarks and 
	# resizes in preparation for AAM training
    path = str(image.path)
    lpath = path.replace('.png','_aam.txt').replace('Images','AAM_landmarks')
    # convert landmark files into menpo landmarkmanager objects and add to image
    lm = landmarkConverter(lpath)
    image.landmarks = lm
    # convert to greyscale
    if image.n_channels == 3:
        image = image.as_greyscale()
    # crop and resize
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image

def prepare_images(folderpath):
	# load and process images from a given directory
	training_paths = []
	training_images = []
	for root, dirs, files in walk(folderpath):
			if len(files) > 0:
				training_paths.append(root)
	for path in training_paths:
		images = mio.import_images(path, verbose=True)
		images = images.map(process)
		training_images += images
	return training_images 

def prepare_labels(trainingimgs):
	# load and process labels from a given directory
	labels = []
	for i in range(len(trainingimgs)):
	    img = trainingimgs[i]
	    path = str(img.path)
	    labelpath = path.replace('Images','Frame_Labels/PSPI').replace('.png','_facs.txt')
	    file = open(labelpath,'r')
	    val = float(file.read())
	    file.close()
	    if val != 0:
	        label = 1
	    else:
	        label = 0
	    labels.append(label)
	    prog = 'image {}'.format(i)
	    print(prog,end='                  \r')
	   # mio.export_pickle(label, destination + 'labels/' + str(i) + '.pkl', overwrite=True)
	print("pain image proportion: {}".format(check_proportions(labels)), end = '             \r')
	return labels

def check_label(image):
	# check if image corresponds to pain classification
	pain = False
	path = str(img.path)
	labelpath = path.replace('Images','Frame_Labels/PSPI').replace('.png','_facs.txt')
	file = open(labelpath,'r')
	val = float(file.read())
	file.close()
	if val!=0:
		pain = True
	return pain

# functions to separate training and test data 
def splitData(folderpath, destination, subject, aamset=True):
	# load and process data into SVM training and test sets, and separate set for AAM training
	# the subject field being filled indicates we are leaving one subject out, rather than using 
	# stratified sampling. The subject to be left out is specified in the subject variable.
	paths = []
	training_images = []
	ptrain =[]
	nptrain = []
	test_paths = []
	test_images = []
	ptest = []
	nptest = []
	AAM_images = []

	# list all folders containing images that are NOT of the test subject
	# make a separate folder of test subject images
	for root, dirs, files in walk(folderpath):
		if len(files) > 0 and subject not in root:
			paths.append(root)
		elif len(files) > 0:
			test_paths.append(root)
	# import training images, divide into training and AAM set (can overlap)
	print("\n\npreparing training images...")
	i = 0
	for path in paths:
		images = mio.import_images(path)
		labels = prepare_labels(images)
		prop = check_proportions(labels)
		print("path {} of {}".format(i, len(paths)), end='         \r')
		if prop > 0:
			images = images.map(process)
			assert len(labels) == len(images)
			vectorp = [n for n in range(len(labels)) if labels[n]==1]
			vectornp = [m for m in range(len(labels)) if labels[m]==0]
			ptrain += images[vectorp]
			nptrain += images[vectornp]
			#train = list(images[vector1])
			#mio.export_pickle(train,destination+'train/imgs/'+str(i)+'.pkl', overwrite=True)
			if aamset:	
				vector3 = [i for i in range(len(labels)) if (labels[i]==1 and i%12==0) or i%31==0]
				AAM_images += images[vector3]
		i+=1

	#import test images, get subset for test
	print("\n\npreparing test images...")
	for path in test_paths:
		images = mio.import_images(path)
		labels = prepare_labels(images)
		prop = check_proportions(labels)
		if prop > 0:
			images = images.map(process)
			vectorp = [i for i in range(len(labels)) if labels[i]==1]
			vectornp = [j for j in range(len(labels)) if labels[j]==0]
			ptest += images[vectorp]
			nptest += images[vectornp]
	print("found {} pain training images, {} no-pain training images".format(len(ptrain),len(nptrain)))
	print("found {} pain test images, {} no-pain test images".format(len(ptest), len(nptest)))
	# get 5000 training images from each, get 100 test images from each 
	ptr = np.linspace(0, len(ptrain)-1, num=5000, dtype=int)
	nptr = np.linspace(0, len(nptrain)-1, num=5000, dtype=int)
	pts = np.linspace(0, len(ptest)-1, num=100, dtype=int)
	npts = np.linspace(0, len(nptest)-1, num=100, dtype=int)
	ptrain = np.asarray(ptrain)
	ptrain = ptrain[ptr]
	nptrain = np.asarray(nptrain)
	nptrain = nptrain[nptr]
	ptest = np.asarray(ptest)
	ptest = ptest[pts]
	nptest = np.asarray(nptest)
	nptest = nptest[npts]
	training_images = np.append(ptrain,nptrain)
	test_images = np.append(ptest,nptest)
	#get corresponding labels for images (AAM images do not need labels)
	training_labels = prepare_labels(training_images)
	test_labels = prepare_labels(test_images)
	print('training set pain proportion: {}'.format(check_proportions(training_labels)))
	print('test set pain proportion: {}'.format(check_proportions(test_labels)))
	# export
	joblib.dump(training_images,destination+'train/imgs/trimgs.pkl')
	mio.export_pickle(training_labels,destination+'train/labels/trl.pkl', overwrite=True)
	mio.export_pickle(test_labels,destination+'test/labels/tstl.pkl', overwrite=True)
	mio.export_pickle(test_images,destination+'test/imgs/tstimgs.pkl', overwrite=True)
	if aamset:
		mio.export_pickle(AAM_images,destination+'aam/aamimgs.pkl', overwrite=True)

def check_proportions(labels):
	# find out how many samples are pain images
	pain = [i for i in labels if i == 1]
	if len(labels) > 0:
		prop = len(pain)/len(labels)
	else:
		prop = None
	return prop

def prune_labels(labels, errors):
	p_labels = [labels[i] for i in range(len(labels)) if i not in errors]
	return p_labels