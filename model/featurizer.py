import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn import metrics
from collections import Counter

def ntuples(arr, n = 2):
    ''' Generator function for subarrays.  Centers never overlap. '''
    for i in range(0, len(arr) - n, n - 1):
        yield arr[i:i+n]

def gen_ngrams(s, n = 2):
    ''' Generator function for ngrams in a sentence represented as a list
        of words.'''
    for i in range(0, len(s)):
        yield s[i:i+n]


def gram_overlap(s1, s2, n = 2):
   
	gram_overlap = []

	for g1 in gen_ngrams(s1, n):
            for g2 in gen_ngrams(s2, n):
                overlap = True
                for elm1, elm2 in zip(g1, g2):
                    if elm1 != elm2:
                        overlap = False
                        break

                    if overlap:
                        gram_overlap.append(' '.join(g1))

        feat = Counter(gram_overlap) 
        feat['{0}gram_overlap'.format(n)] = len(gram_overlap)                      
        return feat


def word_overlap_features(s1, s2):
    overlap = [w1 for w1 in s1 if w1 in s2]

    feat = Counter(overlap)
    feat['overlap_length'] = len(overlap)
    feat['one_not_two_length'] = len([w1 for w1 in s1 if w1 not in s2])
    feat['two_not_one_length'] = len([w2 for w2 in s2 if w2 not in s1])
    
    return feat

def train_logreg(X, y, test_X, test_y, load_vec=True):
	""" 	
	Trains logistic regression on the feature set.
	"""
	full_y = y + test_y
	
	lb = LabelBinarizer()
	lb.fit(full_y)
	# Convert into 1-D array

	model = LogisticRegression()
	big_X = X + test_X

	features = featurize(big_X)
	X, test_X = features[:4500], features[4500:]
	print X.shape, X

	model.fit(X, y)

	y_pred = model.predict(X)
	print set(y_pred)
	print metrics.classification_report(y, y_pred, digits = 3)
	y_pred = model.predict(test_X)
	print set(y_pred)
	print metrics.classification_report(test_y, y_pred, digits = 3)
	# Get Sklearn pipeline here.

def featurize(X):
	"""
	Featurizes using just word overlap features.
	Extracts the unigram features from the given train set.
	- X: numpy array with (n_train, max_seqlen, max_sentlen) 

	Return:
	- vector: A numpy array of size (n_train, n_features)
	corresponding to the featurization of X.  
	"""	
	feats = []

	n_train = len(X)
	print n_train
	for i in xrange(0, n_train, 2):
		feat_dict = {}
		s1 = X[i]
		s2 = X[i + 1]
		feat_dict.update(word_overlap_features(s1, s2))
		feats.append(feat_dict)
		    
	v = DictVectorizer(sparse=True)
	vector = v.fit_transform(feats)

	return vector
