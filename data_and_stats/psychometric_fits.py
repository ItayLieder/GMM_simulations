import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from scipy.optimize import minimize

def phi(x,mu=0,sd=1):
    """
    Cumulative Gaussian function evaluated at x for parameters mu, sd
    """
    return 0.5 * (1 + sc.special.erf((x - mu) / (sd * np.sqrt(2))))

def get_fit_grid(df,y,typ=0,bounds=[-0.05,0.05,0,0.3],grid=[60,60]):
    """
    Likelihood map for psychometric function
    :param df: frequency difference (input)
    :param y: binary decision
    :return:
    """
    assert len(df.shape) == 1
    assert len(y.shape) == 1
    assert typ in [0,1,2]
    assert y.shape == df.shape
    n_s = grid[0]
    n_a = grid[1]
    s_a = np.linspace(bounds[0],bounds[1],n_s)
    s_s = np.linspace(bounds[2],bounds[3],n_a)

    eps = 1.e-5
    if typ == 0:
        S_A,S_S,DF = np.meshgrid(s_a,s_s,df)
        _,_,Y = np.meshgrid(s_a,s_s,y)
        T = phi(DF,S_A,S_S)
        T[T<eps]=eps
        T[T>1-eps]=1.-eps
        ll = np.sum(Y*np.log(T)  + (1-Y)*np.log(1.-T),axis=2)
    elif typ == 1:
        S_A,S_S = np.meshgrid(s_a,s_s)
        eps = 1.e-5
        ll = np.zeros((n_s,n_a))
        for y_,df_ in zip(y,df):
            T = phi(df_,S_A,S_S)
            T[T<eps]=eps
            T[T>1-eps]=1.-eps
            ll += y_*np.log(T)  + (1-y_)*np.log(1.-T)
    elif typ==2:
        ll = np.zeros((n_s,n_a))
        for ii,sigma in enumerate(s_s):
            for jj,alpha in enumerate(s_a):
                t= phi(df,alpha,sigma)
                eps = 1.e-5
                t[t<eps]=eps
                t[t>1-eps]=1.-eps
                ll[ii,jj] = np.sum( y*np.log(t)  + (1-y)*np.log(1.-t) );


    normalizer = sc.misc.logsumexp([ll])

    ll = np.exp(ll  - normalizer)

    return s_s,\
            s_a,\
            ll

def llh_bern(x,y,mu,sd):
    B = phi(x,mu=mu,sd=sd)
    eps = 1.e-6
    B[B<eps]=eps
    B[B>1-eps]=1-eps
    return np.sum(  y*np.log(B)+(1-y)*np.log(1-B)  )

def llh(x,y,mu,sd):
    B = phi(x,mu=mu,sd=sd)
    eps = 1.e-6
    B[B<eps]=eps
    B[B>1-eps]=1-eps
    return  np.sum(  y*np.log(B)+(1-y)*np.log(1-B)  )


def ml_fit_psych(x,y,mu_0=0.,sd_0=1.):
    f = lambda p: -llh(x,y,p[0],p[1])
    res = minimize(f,[mu_0,sd_0], method='L-BFGS-B')
    return res.x,res.fun


def plot_psychometric_parameters(Fc,prm,name=None,axarr=None):
    """
    for center frequency Fc, plots parameters alpha,sigma
    :param Fc:
    :param prm:
    :param name:
    :param axarr:
    :return:
    """
    if axarr==None:
        fig,axarr = plt.subplots(2,1,figsize=(8,6))
    axarr[0].plot(Fc,prm[:,0],label=name)
    axarr[0].set_title('alpha')
    axarr[1].plot(Fc,prm[:,1],label=name)
    axarr[1].set_title('sigma')
    axarr[1].set_ylim([0,0.3])

    for ax in axarr:
        ax.set_xlim([Fc.min(),Fc.max()])
        ax.set_xlabel('f')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
    return axarr



