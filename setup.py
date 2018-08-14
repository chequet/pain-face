import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import menpo.io as mio
from conversion import landmarkConverter
from os import walk
import numpy as np

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
	    print(prog,end='       \r')
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
def splitData(folderpath, destination, subject=None):
	# load and process data into SVM training and test sets, and separate set for AAM training
	# the subject field being filled indicates we are leaving one subject out, rather than using 
	# stratified sampling. The subject to be left out is specified in the subject variable.
	paths = []
	training_images = []
	test_paths = []
	test_images = []
	# AAM_paths = []
	AAM_images = []

	if subject == None:
		# list all folders containing images
		for root, dirs, files in walk(folderpath):
			if len(files) > 0:
				paths.append(root)
		# go through image folders and import images
		for path in paths:
			images = mio.import_images(path, verbose=True)
			images = images.map(process)
			# divide into training, test and aam -- 
			# remove from training and aam anything in test
			vector1 = np.arange(0,len(images),4)
			vector2 = np.arange(0,len(images),80)
			vector3 = np.arange(0,len(images),31)
			vector1 = [num for num in vector1 if num not in vector2]
			vector3 = [num for num in vector3 if num not in vector2]
			# add images to appropriate lists
			#training_images += images[vector1]
			train = images[vector1]
			mio.export_pickle(train, destination+'train/imgs/')
			test_images += images[vector2]
			AAM_images += images[vector3]
	else:
		# list all folders containing images that are NOT of the test subject
		# make a separate folder of test subject images
		for root, dirs, files in walk(folderpath):
			if len(files) > 0 and subject not in root:
				paths.append(root)
			elif len(files) > 0:
				test_paths.append(root)
		# import training images, divide into training and AAM set (can overlap)
		print("\n\npreparing training images...")
		i=0
		for path in paths:
			images = mio.import_images(path)
			labels = prepare_labels(images)
			prop = check_proportions(labels)
			if prop > 0:
				images = images.map(process)
				assert len(labels) == len(images)
				vector1 = [i for i in range(len(labels)) if labels[i]==1 or i%5==0]
				vector3 = [i for i in range(len(labels)) if (labels[i]==1 and i%12==0) or i%31==0]
				training_images += images[vector1]
				train = list(images[vector1])
				mio.export_pickle(train,destination+'train/imgs/'+str(i)+'.pkl', overwrite=True)
				AAM_images += images[vector3]
				i+=1
		#import test images, get subset for test
		print("\n\npreparing test images...")
		j=0
		for path in test_paths:
			images = mio.import_images(path)
			labels = prepare_labels(images)
			prop = check_proportions(labels)
			if prop > 0:
				images = images.map(process)
				vector2 = [i for i in range(len(labels)) if labels[i]==1 or i%10==0]
				test_images += images[vector2]
				j+=1
	#get corresponding labels for images (AAM images do not need labels)
	training_labels = prepare_labels(training_images)
	test_labels = prepare_labels(test_images)
	mio.export_pickle(training_labels,destination+'train/labels/trl.pkl', overwrite=True)
	mio.export_pickle(test_labels,destination+'test/labels/tstl.pkl', overwrite=True)
	mio.export_pickle(test_images,destination+'test/imgs/tstimgs.pkl', overwrite=True)
	mio.export_pickle(AAM_images,destination+'aam/aamimgs.pkl', overwrite=True)
	return tr, trl, tst, tstl, aam

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