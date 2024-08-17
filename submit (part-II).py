import numpy as np
from sklearn import svm
from scipy.linalg import khatri_rao
import time as tm
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
from sklearn.linear_model import LogisticRegression

params_ = []

lda_clf=None

def my_map( X ):
    if (lda_clf):
        return lda_clf.transform(X)
    return None

def my_fit( X_train, y_train ):
    feat = my_map_(X_train)
    lda_clf = lda().fit(feat[:10000], y_train[:10000])
    ft = my_map(feat)
    clf = LogisticRegression().fit(ft, y_train)
    w, b = clf.coef_.T.flatten(), clf.intercept_
    return w, b

def my_map_( X ):
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

    return np.array(feat)


def find_acc(feat, w, b):
    scores = feat.dot( w ) + b
    pred = np.zeros_like( scores )
    pred[scores > 0] = 1
    acc = np.average( Z_tst[ :, -1 ] == pred )
    return acc

if __name__ == "__main__":
    Z_trn = np.loadtxt( "train.dat" )
    Z_tst = np.loadtxt( "test.dat" )

    
    n_trials = 5

    d_size = 0
    t_train = 0
    t_trainl = 0
    t_map = 0
    acc = 0
    acc_l = 0
    for t in range( n_trials ):
        tic = tm.perf_counter() 
        w, b = my_fit( Z_trn[:, :-1], Z_trn[:,-1] )
        toc = tm.perf_counter()
        t_train += toc - tic
        d_size += w.shape[0]
        tic = tm.perf_counter()
        feat = my_map( Z_tst[:, :-1] )
        toc = tm.perf_counter()
        t_map += toc - tic
        scores = feat.dot( w ) + b
        pred = np.zeros_like( scores )
        pred[scores > 0] = 1
        acc += np.average( Z_tst[ :, -1 ] == pred )
    d_size /= n_trials
    t_train /= n_trials
    t_trainl /= n_trials
    t_map /= n_trials
    acc /= n_trials
    acc_l /= n_trials

    print( d_size, t_map)
    print(t_train, 1-acc)

        
    
    '''
    d_size = 0
    t_train_l = []
    t_train_s = []
    t_map = 0
    ft_time = 0
    acc_l = []
    acc_s = []
    
    tic = tm.perf_counter()
    feat = my_map( Z_tst[:, :-1] )
    toc = tm.perf_counter()
    t_map = toc - tic
    
    tic = tm.perf_counter()
    ft = my_map( Z_trn[:, :-1] )
    toc = tm.perf_counter()
    ft_time = toc - tic
    
    tol = []
    
    for i in range(1, 7):
        tol.append(10**(-i))
        tic = tm.perf_counter()
        clf = LogisticRegression(C=6.0, tol = tol[-1]).fit(ft, Z_trn[:, -1])
        w, b = clf.coef_.reshape(528), clf.intercept_
        toc = tm.perf_counter()
        t_train = toc-tic
        t_train += ft_time
        t_train_l.append(t_train)

        acc_l.append(find_acc(feat, w, b))

        tic = tm.perf_counter()
        clf = svm.LinearSVC(dual=False, C=2.7, tol=tol[-1]).fit(ft, Z_trn[:, -1])
        w, b = clf.coef_.reshape(528), clf.intercept_
        toc = tm.perf_counter()
        t_train = toc-tic
        t_train += ft_time
        t_train_s.append(t_train)

        acc_s.append(find_acc(feat, w, b))

    d_size=feat.shape[1]
    print("Dimensions =", d_size, "Map time =", t_map)
    t_s = np.array(t_train_s)
    t_l = np.array(t_train_l)
    t = np.array(tol)
    plt.plot(t, t_s)
    plt.plot(t, t_l)
    plt.xscale("log")
    plt.show()
    plt.plot(t, np.array(acc_s))
    plt.plot(t, np.array(acc_l))
    plt.xscale("log")
    plt.show()
    '''

    
    
