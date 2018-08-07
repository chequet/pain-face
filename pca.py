import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from AAM import generate_capp, AAMfitter, generate_normcapps
from setup import splitData
from sklearn import svm
import numpy as np
import menpo.io as mio

trl = mio.import_pickle('svm_trainlab.pkl')
tstl = mio.import_pickle('svm_tstlab.pkl')

traincapps = mio.import_pickles('~/pain-face-master/traincapps/')
testcapps = mio.import_pickles('~/pain-face-master/testcapps/')
trainerrs = mio.import_pickles('/home/ch283/pain-face-master/trainerrs/')
testerrs = mio.import_pickles('/home/ch283/pain-face-master/testcapps/errs/')

tstl2 = tstl[2000:3097]
trainlabels = [trl[i] for i in range(len(trl)) if i not in trainerrs]
testlabels = [tstl2[i] for i in range(len(tstl2)) if i not in testerrs]

from sklearn.decomposition import IncrementalPCA
pca = IncrementalPCA(n_components=30, batch_size=50)
tc = pca.fit_transform(traincapps)


from sklearn.svm import LinearSVC
clf = svm.LinearSVC()
clf.fit(tc,trainlabels)
tsc = pca.transform(testcapps[500:1000])
pred = clf.predict(tsc)
from sklearn.metrics import accuracy_score
accuracy_score(testlabels[500:1000],pred)