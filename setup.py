import menpo.io as mio
from conversion import landmarkConverter
from os import walk

def process(image, crop_proportion=0.2, max_diagonal=400):
    path = str(image.path)
    lpath = path.replace('.png','_aam.txt').replace('Images','AAM_landmarks')
    lm = landmarkConverter(lpath)
    image.landmarks = lm
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image

def prepare_images(folderpath):
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