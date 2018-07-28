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