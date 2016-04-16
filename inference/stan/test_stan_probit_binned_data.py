import numpy as np
import pickle
import pystan
from matplotlib import pyplot as plt
from scipy.io import loadmat
from pystan import StanModel


fname = '/nfs/nhome/live/vincenta/git/io-pitch-discrimination/code/python/stan/wideRange.mat'
data = loadmat(fname)


#---------------------------------- Filtering by acc
Isub =[]
for i_sub in range(58):
    Acc=data['acc'][i_sub,:]
    if Acc.mean()>0.8 and Acc.mean()<1.:
        Isub.append(i_sub)
print Isub

#---------------------------------- Arranging data

F1 = np.log(np.array(data['s1'][Isub,:]))
F2 = np.log(np.array(data['s2'][Isub,:]))
Y = data['resp'][Isub,:]
Acc=data['acc'][Isub,:]


df = (F1 - F2)
d1 = np.zeros(df.shape)
d2 = np.zeros(df.shape)
d1[:,1:] = F1[:,1:]-0.5*(F1+F2)[:,:-1]
d2[:,2:] = F1[:,2:]-0.5*(F1+F2)[:,:-2]

d1 = d1[:,2:].flatten()
d2 = d2[:,2:].flatten()
df = df[:,2:].flatten()
y = Y[:,2:].astype(int).flatten()
N = len(df)


#--------------------------------
# Binning Data
o1 = np.argsort(d1)
d1=d1[o1];d2=d2[o1];df=df[o1];y=y[o1]
w1 = 0.1*(d1.max()-d1.min())
w2 = 0.1*(d2.max()-d2.min())

bin_edges = np.linspace(d1.min(),d1.max(),10)
bins = [[el,eu] for el,eu in zip(bin_edges[:-1],bin_edges[1:]) ]
bins1 = [[s,s+w1] for s in np.linspace(d1.min(),d1.max()-w1,50) ]
bins2 = [[s,s+w2] for s in np.linspace(d2.min(),d2.max()-w2,50) ]

binsc = [np.mean(b) for b in bins]
binsc1 = [np.mean(b) for b in bins1]
binsc2 = [np.mean(b) for b in bins2]




binsc2d = []
bins2d = []
Is = []
for b1 in bins1:
    for b2 in bins2:
        binsc2d.append([np.mean(b1),np.mean(b2)])
        bins2d.append([b1,b2])
        print b1,b2
        i = np.where(  (b1[0]<d1) & (d1<b1[1]) & (b2[0]<d2) & (d2<b2[1])   )[0]
        count = len(i)
        Is.append(
            np.where(  (b1[0]<d1) & (d1<b1[1]) & (b2[0]<d2) & (d2<b2[1])   )[0]
        )

#Is = [np.where( ((el<d1) & (d1<eu)))[0] for el,eu in bins]
#Is = [np.where( (el<d1) & (d1<eu) & (el<d2) & (d2<eu) )[0] for (el1,eu1),(el2,eu2) in zip(bins1,bins2)]

print [len(I) for I in Is]
#--------------------------------

model = 'simple'
#sm = StanModel(file='psych_model.stan')
#with open(model+'_model.pkl','wb') as f:
#    pickle.dump(sm, f)
sm = pickle.load(open(model+'_model.pkl','rb'))

res =[]
for I in Is:

    n=len(I)
    ext = []
    if n>10:
        print n,y[I].shape,df[I].shape
        model_dat = {'N': n,
                   'y': y[I],
                   'df': df[I]}
        fit = sm.sampling(data=model_dat)
        ext = fit.extract()
    res.append(ext)


pickle.dump({'res':res,'binsc':binsc2d}, open('bin_save_fit.p','wb'))


print Acc.mean()
print 'df:',df.min(),df.max(), 'range:',df.max()-df.min()
print df.shape,d1.shape



