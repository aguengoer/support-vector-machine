import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


def MMC(features,targets):
    for ind,tar in enumerate(targets):
        if tar == 0:
            targets[ind]=-1
    temp_features=pd.DataFrame(features)
    M_max=0
    phi_m_max=-10000
    offset_best=-100000000000000
    offs_range=np.linspace(-10,10,100)
    x2_min=-2
    x1_best=-2
    for x1 in np.linspace(-1,1,1000):
        if x1**2 ==1: x2_1,x2_2=[0,0]
        else:
            x2_1=np.sqrt(1-x1**2)
            x2_2 = -np.sqrt(1 - x1**2)
        for offset in offs_range:
            #offset=0
            M_min_1 = 10000000000000
            M_min_2 = 10000000000000
            x2_1_flg = True
            x2_2_flg = True
            for ind,feat in enumerate(features):
                #print((x1*feat[0]+x2_1*feat[1]+offset),targets[ind],feat)

                if x2_1_flg:
                    if ((x1*feat[0]+x2_1*feat[1]+offset)*targets[ind])<0:
                        x2_1_flg=False
                        M_min_1 = 10000000000000
                    else:
                        M=((x1*feat[0]+x2_1*feat[1]+offset)*targets[ind])
                        #print(M)
                        if M < M_min_1:
                            M_min_1=M
                            x2_min=x2_1
                            offset_best=offset

                if x2_2_flg:
                    #print(((x1*feat[0]+x2_2*feat[1]+offset)*targets[ind]))
                    if ((x1*feat[0]+x2_2*feat[1]+offset)*targets[ind])<0:
                        x2_2_flg=False
                        M_min_2=10000000000000
                    else:
                        M =((x1*feat[0]+x2_2*feat[1]+offset)*targets[ind])
                        #print(M)
                        if M < M_min_2:
                            M_min_2=M
                            x2_min = x2_2
                            offset_best=offset
                if not (x2_1_flg or x2_2_flg):
                    break

            if M_min_1>M_max and M_min_1 !=10000000000000:
                M_max=M_min_1
                x1_best=x1
                x2_best=x2_min
                offset_best_best=offset_best
                #print(M_max,x1_best,x2_best,offset_best_best)
            if M_min_2>M_max and M_min_2 !=10000000000000:
                M_max=M_min_2
                x1_best=x1
                x2_best=x2_min
                offset_best_best=offset_best
                #print(M_max,x1_best,x2_best,offset_best_best)





    x_range=np.linspace(temp_features.iloc[:,0].min(),temp_features.iloc[:,0].max(),1000)
    y_range = -x1_best/x2_best * x_range-offset_best_best/x2_best

    #plt.scatter(features[:, 0], features[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    plt.plot(x_range,y_range)
    #plt.show()
    return phi_m_max

if __name__=="__main__":
    # we create 40 separable points
    X, y = make_blobs(n_samples=40, centers=2, random_state=6)
    X, y = make_blobs(n_samples=40, centers=2)

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(X, y)
    phi_ideal=MMC(X,y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(
        XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
    )
    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.show()




