import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
from DataLoader import *
from utils import *

# --------------------------------

def trial_sorting_global(F1,F2,a,b,mu_inf, alpha=0.5):
    """
    applies to unimodal data only
    Sorts trials into 4 categories as above or below mean of generative distribution
    and close or far from this mean
    :return: indices of trials in each category (up/ddwn)*(far/close)
    """

    th =  alpha*(b-a)# what distance from mu_inf to cut for close/far trials
    g_up = (F1>mu_inf)&(F2>mu_inf)
    g_down = (F1<mu_inf)&(F2<mu_inf)
    locd_c = g_down & ((F1>mu_inf -th)&(F2>mu_inf -th))
    locd_f = g_down & ((F1<mu_inf-th)&(F2<mu_inf-th))
    locu_f = g_up & ((F1>mu_inf+th)&(F2>mu_inf+th))
    locu_c = g_up & ((F1<mu_inf+th)&(F2<mu_inf+th))
    return locd_f,locd_c,locu_f,locu_c


def trial_sorting_p_m(F1,F2):
    """ Trial sorting bias + / bias - as in Ofri's Pcb
    :return: indices of trials in bias - and bias + regions
    """
    mu_inf = 0.5*(np.mean(F1)+np.mean(F2))
    Im = (F1>mu_inf)&(F1>F2) | (F1<mu_inf)&(F2>F1)
    Ip = (F1>mu_inf)&(F1<F2) | (F1<mu_inf)&(F2<F1)
    return Im,Ip


def trial_sorting_local(F1,F2,a,b,mu_inf,alpha=0.5):
    """ Similar to trial sorting global but trials are now separated with respect to average of previous frequencies at t-1
    :return: indices of trials in each category (up/ddwn)*(far/close)
    """
    n_s,n_t = F1.shape
    mu_prev = np.ones((n_s,n_t))*mu_inf
    for i_s in range(n_s):
        mu_prev[i_s,:-1] = 0.5*(F1[i_s,1:]+F2[i_s,1:]) # mean of the last previous tones

    th =  alpha*(b-a) # what distance from mu_prev to cut

    g_up = (F1>mu_prev)&(F2>mu_prev)
    g_down = (F1<mu_prev)&(F2<mu_prev)
    locd_c = g_down & ((F1>mu_prev-th)&(F2>mu_prev-th))
    locd_f = g_down & ((F1<mu_prev-th)&(F2<mu_prev-th))
    locu_f = g_up & ((F1>mu_prev+th)&(F2>mu_prev+th))
    locu_c = g_up & ((F1<mu_prev+th)&(F2<mu_prev+th))
    return locd_f,locd_c,locu_f,locu_c




def sliding_window_indices(F1,F2,mu_p,w_f=0.2,overlap=0.5):
    assert len(F1.shape)==1
    assert len(F2.shape)==1
    Fmr,ir = rotate(F1,F2,c=[mu_p,mu_p],t=np.pi/4.)
    fmin = Fmr.min()
    fmax = Fmr.max()
    w_width = w_f*(fmax-fmin)
    Fc = np.linspace(fmin+w_width/2, fmax-w_width/2, 1./w_width/(1-overlap))
    edges = np.array([[ fc-w_width/2,fc+w_width/2] for fc in Fc])
    indices = [np.where( (el<Fmr)&(eu>=Fmr) ) for el,eu in edges]
    return indices,Fc



# --------------------------------

def accuracy(F1,F2,Y):
    ''' Computes the accuracy of subject responses
    :return: accuracy
    '''
    if len(Y.shape)==2:
        n_s,n_t = F1.shape
        acc = np.zeros((n_s,))
        for i_s in range(n_s):
            f1 = F1[i_s,:]
            f2 = F2[i_s,:]
            y=Y[i_s,:]
            acc[i_s] = np.mean(( (f1-f2>0)&y )|( (f1-f2<0)&~y ))
    else:
        acc= np.mean(( (F1-F2>0)&Y )|( (F1-F2<0)&~Y ))
    return acc

def bias_p_m(F1,F2,Y):
    """ computes the accuracy per region (bias + or bias -)
    :return: accuracy in bias + and bias - regions
    """
    assert len(Y.shape) == 2
    n_s,n_t = F1.shape
    bias = np.zeros((n_s,2))
    Im,Ip = trial_sorting_p_m(F1,F2)
    for i_s in range(n_s):
        im= Im[i_s,:]
        ip= Ip[i_s,:]
        bias[i_s,:] = [ accuracy(F1[i_s,:][im],F2[i_s,:][im],Y[i_s,:][im]),\
                        accuracy(F1[i_s,:][ip],F2[i_s,:][ip],Y[i_s,:][ip]) ]
    return bias



#----------------------------------

#------------------------------------------



