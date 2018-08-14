import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from sklearn.neural_network import MLPClassifier
import menpo.io as mio
import joblib
from sklearn.metrics import accuracy_score, classification_report

def trainshallow(destination, PCA=True):
	# get CAPPs or PCA model of CAPPs
	if PCA:
		traincapps = joblib.load(destination+'train/pca/trainpca.pkl')
		testcapps = joblib.load(destination + 'test/pca/testpca.pkl')
	else:
		traincapps = mio.import_pickles(destination +'train/')
		testcapps = mio.import_pickles(destination + 'test/')

	# get labels
	trainlabels = mio.import_pickle(destination + 'train/labels/trl_pruned.pkl')
	testlabels = mio.import_pickle(destination + 'test/labels/tstl_pruned.pkl')

	# create and train network 
	clf = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
	clf.fit(traincapps,trainlabels)
	joblib.dump(clf, destination+'NN/shallow/clf.pkl')

	# make a prediction
	pred = clf.predict(testcapps)
	joblib.dump(pred, destination+'NN/shallow/pred.pkl')

	# check prediction
	print(accuracy_score(testlabels,pred))
	class_names = ['no pain', 'pain']
	print(classification_report(testlabels,pred, target_names=class_names))
