import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import menpo.io as mio
from conversion import landmarkConverter
from setup import process, prepare_images
from os import walk
from menpofit.aam import HolisticAAM
from menpo.feature import fast_dsift
import numpy as n

def generate_capp(data-path):
	training_images = prepare_images(data-path)

	aam = HolisticAAM(training_images, group='landmarks', diagonal=150,
                  scales=(0.5, 1.0), holistic_features=fast_dsift, verbose=True,
                  max_shape_components=20, max_appearance_components=150)
	app = aam.appearance_models[-1]
	capp = app.mean()
	return capp.as_vector()