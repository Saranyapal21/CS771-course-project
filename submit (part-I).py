import numpy as np
import sklearn
from scipy.linalg import khatri_rao

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to train your model using training CRPs
    # X_train has 32 columns containing the challeenge bits
    # y_train contains the responses

    # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
    # If you do not wish to use a bias term, set it to 0
    from sklearn.linear_model import LogisticRegression
    feat = my_map(X_train)
    clf = LogisticRegression(dual=False, C=6.0).fit(feat, y_train)
    w, b = clf.coef_.T.flatten(), clf.intercept_
    return w, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to create features.
    # It is likely that my_fit will internally call my_map to create features for train points
    X = 2*X-1
    n=len(X)
    n_=len(X[0])
    X = np.flip(np.cumprod(np.flip(X, axis=1), axis=1), axis=1)
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    n=len(X)
    m=len(X[0])
    feat = np.empty((n, int(m*(m-1)/2)), dtype = X.dtype)
    ind = 0
    for i in range(m):
        for j in range(i+1, m):
            feat[:, ind] = 2 * X[:, i] * X[:, j]
            ind+=1
    return feat
