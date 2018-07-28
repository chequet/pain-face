import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from setup import prepare_images, process
import numpy as np
import menpo.io as mio
from menpofit.aam import HolisticAAM
from menpo.feature import fast_dsift
from menpofit.aam import LucasKanadeAAMFitter, WibergInverseCompositional
from menpodetect import load_dlib_frontal_face_detector

def AAMfitter(training_path):
	# load and prepare training images
	training_images = prepare_images(training_path)
	# train AAM
	print('Now generating Active Appearance Model...')
	aam = HolisticAAM(training_images, group='landmarks', diagonal=150,
                  scales=(0.5, 1.0), holistic_features=fast_dsift, verbose=True,
                  max_shape_components=20, max_appearance_components=150)
	# generate AAM fitter
	print('Now generating fitter...')
	fitter = LucasKanadeAAMFitter(aam)
	return fitter, training_images

def generate_capp(fitter, image):
	print('generating canonical appearance of file ', image.path)
	# NB to save time we assume image is already preprocessed
	# Load detector for face area
	detect = load_dlib_frontal_face_detector()
	# Detectbounding box
	bboxes = detect(image)
	# initial bbox
	initial_bbox = bboxes[0]
	# fit image
	result = fitter.fit_from_bb(image, initial_bbox, max_iters=[15, 5],
                            gt_shape=image.landmarks['landmarks'].lms)
	# get CAPP from fitter object
	capp = fitter.appearance_reconstructions(result.appearance_parameters,result.n_iters_per_scale)[-1]
	capp = capp.as_vector()
	return capp

def generate_normcapps(fitter, training_images):
	print("Generating canonical appearance vectors...")
	# again assume image is already preprocessed
	capps = []
	for img in training_images:
		capp = generate_capp(fitter, image)
		# normalise this for classifier input
		capp = capp/(np.max(capp))
		capps.append(capp)
	return capps