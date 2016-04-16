import numpy as np
import pickle
import pystan
from matplotlib import pyplot as plt
from scipy.io import loadmat
from pystan import StanModel


fname = '/nfs/nhome/live/vincenta/git/io-pitch-discrimination/code/python/stan/wideRange.mat'
data = loadmat(fname)



df = (F1 - F2)
d1 = np.zeros(df.shape)
d2 = np.zeros(df.shape)
d1[1:] = F1[1:]-0.5*(F1+F2)[:-1]
d2[2:] = F1[2:]-0.5*(F1+F2)[:-2]

n=100
d1 = d1[2:n]
d2 = d2[2:n]
df = df[2:n]
y = Y[2:n].astype(int)

N = len(df)


print Acc.mean()
print 'df:',df.min(),df.max(), 'range:',df.max()-df.min()
print 'd1:',d1.min(),d1.max(), 'range:',d1.max()-d1.min()


model = 'gp_bias'

if model == 'simple':
    #sm = StanModel(file='psych_model.stan')
    #with open(model+'_model.pkl','wb') as f:
    #    pickle.dump(sm, f)
    sm = pickle.load(open(model+'_model.pkl','rb'))
    model_dat = {'N': N,
               'y': y,
               'df': df}
elif model == 'gp_bias':

    #sm = StanModel(file='model.stan')
    #with open(model+'_model.pkl','wb') as f:
    #    pickle.dump(sm, f)
    sm = pickle.load(open(model+'_model.pkl','rb'))

    beta1=1
    beta2=(0.7)**2
    beta3=0.001
    model_dat = {'N': N,
               'y': y,
               'df': df,
               'd1':d1,
               'd2':d2,
               'beta1':beta1,
               'beta2':beta2,
               'beta3':beta3}



print df.shape,d1.shape






fit = sm.sampling(data=model_dat)

ext = fit.extract()

d={'ext':ext,'model_dat':model_dat}
pickle.dump(d, open(model+'_save_fit.p','wb'))
