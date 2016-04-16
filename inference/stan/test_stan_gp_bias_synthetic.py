import numpy as np
import pickle
import pystan
from matplotlib import pyplot as plt
print 'test'


from pystan import StanModel

sm = StanModel(file='model.stan')
with open('model.pkl','wb') as f:
    pickle.dump(sm, f)

import scipy as sc
def phi(x,mu=0,sd=1):
    return 0.5 * (1 + sc.special.erf((x - mu) / (sd * np.sqrt(2))))

N = 300
df = 2*(np.random.random(N)-0.5)*0.6
d1 = 2*(np.random.random(N)-0.5)
A = lambda sc,de : lambda d : sc*d*np.exp(-np.abs(d)/de) 
alpha = A(5,1)
sigma = 0.06
y = (np.random.rand(N)<phi(df/sigma-alpha(d1))).astype(int)

plt.plot(d1,alpha(d1),'rx',)
plt.plot(d1,df/sigma,'gx',)


sm = pickle.load(open('model.pkl','rb'))

beta1=2.
beta2=2.
beta3=0.001
cauchy=0.1
model_dat = {'N': N,
               'y': y,
               'df': df,
               'd1':d1,
                'beta1':beta1,'beta2':beta2,'beta3':beta3,
              'cauchy':cauchy}

fit = sm.sampling(data=model_dat)#,algorithm="Fixed_param")

ext = fit.extract()

data = {'df':df,'d1':d1}
d={'fit':fit,'model':sm,'model_dat':model_dat}
pickle.dump(d, open('save_fit.p','wb'))
