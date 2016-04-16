"""
Functions to plot 2 tone discrimination data
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors as NN
from matplotlib import pyplot as plt



def plot_f1_f2_kde(F1,F2,Y,r=100.,n_g=50):
    """
    Plotting kernel density estimate of response probability from binary decisions
    :param F1, F2, Y: stimuli and responses
    :param r: scale of gaussian kernel used (fraction of range of frequencies)
    :param n_g: grid size
    """
    f1min = F1.min()
    f1max = F1.max()
    f2min = F2.min()
    f2max = F2.max()
    s = (f1max-f1min)/r # scale
    x1,x2 = np.meshgrid(np.linspace(f1min,f1max,n_g),\
                              np.linspace(f2min,f2max,n_g))
    n_t = len(F1)
    I = (Y==1)
    W = np.zeros((n_g,n_g,n_t))

    for f1,f2,y,i_t in zip(F1,F2,Y,range(n_t)):
        W[:,:,i_t] = np.exp(-0.5/s**2*((x1-f1)**2+(x2-f2)**2))
    SW = np.sum(W,axis=2) # sum over trials

    # Normalizing
    for i_t in range(n_t):
        W[:,:,i_t]/=SW

    # Final map
    M = np.sum(W[:,:,I],axis=2)

    fig,ax = plt.subplots()
    cax = ax.imshow(M,origin='lower',extent=[f1min,f1max,f2min,f2max],\
              cmap=plt.cm.coolwarm,\
              interpolation='nearest',aspect='auto')
    ax.axhline(y=0,c='k')
    ax.set_xlabel('$f_1$')
    ax.set_ylabel('$f_2$')
    ax.set_title('% f1 higher')
    fig.colorbar(cax, ax=ax)
    #plt.show()

def plot_f1_f2_hxbin(F1,F2,Y,gridsize=50):
    """
    Hexbin plot
    :param F1, F2, Y: stimuli and responses
    :param gridsize: number of hexbins
    """
    plt.hexbin(F1[Y==1],F2[Y==1],gridsize=gridsize)
    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.title('% f1 higher')
    plt.colorbar()
    #plt.show()


def plot_f1_f2_NN_resp(F1,F2,Y,nn=3,**kwargs):
    """
    Nearest neighbour estimation of response probability
    :param F1, F2, Y: stimuli and responses
    :param nn: number of nearest neighbours
    :param kwargs: any additional arguments of plt.scatter
    :return:
    """
    y = Y.flatten()
    f1 = F1.flatten()
    f2= F2.flatten()
    pairs = np.vstack((f1,f2))
    nbrs = NN(n_neighbors=nn, algorithm='ball_tree').fit(pairs.T)
    distances, indices = nbrs.kneighbors(pairs.T)
    avNN = [np.mean(y[ind]) for ind in indices]
    avNN-=np.mean(avNN)
    avNN/=np.std(avNN)
    plt.scatter(f1,f2,c=avNN,marker='o',cmap=plt.cm.coolwarm,**kwargs)
    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.title('% f1 higher')
    plt.colorbar()
    #plt.show()