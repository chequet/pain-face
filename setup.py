import menpo.io as mio
from conversion import landmarkConverter
from os import walk

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
	for img in trainingimgs:
	    path = str(img.path)
	    labelpath = path.replace('Images','Frame_Labels/PSPI').replace('.png','_facs.txt')
	    file = open(labelpath,'r')
	    val = float(file.read())
	    if val != 0:
	        label = 1
	    else:
	        label = 0
	    labels.append(label)
	return labels

# functions to separate training and test data 

def splitData(folderpath, subject=None):
	# load and process data into training and test sets
	# the subject field being filled indicates we are leaving one subject
	# out, rather than using stratified sampling. The subject to be left out 
	# is specified in the subject path
	training_paths = []
	training_images = []
	test_paths = []
	test_images = []

	if subject == None:
		# add every tenth file to test set
		i = 0;
		for root, dirs, files in walk(folderpath):
			if len(files) > 0 and i%10!=0:
				training_paths.append(root)
				#print(i,'appending to training', root)
				i += 1
			elif len(files) > 0:
				test_paths.append(root)
				#print(i, 'appending to test', root)
				i += 1 
	else:
		# add all files that aren't in left out subject to training,
		# add left out subject to test
		for root, dirs, files in walk(folderpath):
			if len(files) > 0 and subject not in root:
				training_paths.append(root)
			elif len(files) > 0:
				test_paths.append(root)

	for path in training_paths:
		images = mio.import_images(path, verbose=True)
		images = images.map(process)
		training_images += images
	for path in test_paths:
		images = mio.import_images(path, verbose=True)
		images = images.map(process)
		test_images += images
	training_labels = prepare_labels(training_images)
	test_labels = prepare_labels(test_images)

	return training_images, training_labels, test_images, test_labels