import numpy as np
import pickle
import pystan
from matplotlib import pyplot as plt
%matplotlib inline
print 'test'

from pystan import StanModel

model_code = """
data {
    int<lower=1> N;
    vector[N] df;
    vector[N] d1;
    int<lower=0,upper=1> y[N];
    real<lower=0> beta1;
    real<lower=0> beta2;
    real<lower=0> beta3;
}

transformed data {
    vector[N] mu; // Prior mean of the process
    mu <- rep_vector(0, N);
}

parameters {
    vector[N] alpha;
    real<lower=0> sigma;
}
model {

    matrix[N, N] Sigma;
    
    for(i in 1:N)
    for(j in i:N){
        Sigma[i, j] <- beta1 * exp(-beta2 * pow(d1[i] - d1[j], 2)) 
            + if_else(i==j, beta3, 0.0);
    }

    for(i in 1:(N-1))
        for(j in (i+1):N){
            Sigma[j, i] <- Sigma[i, j];
    }

    alpha ~ multi_normal(mu, Sigma);
    sigma ~ cauchy(0,0.5);
    for (i in 1:N)
        {
            y[i] ~ bernoulli( Phi( df[i] / sigma - alpha[i] ) );
        }

}"""

sm = StanModel(model_code=model_code)
with open('model.pkl','wb') as f:
    pickle.dump(sm, f)


import scipy as sc
def phi(x,mu=0,sd=1):
    return 0.5 * (1 + sc.special.erf((x - mu) / (sd * np.sqrt(2))))

N = 30
df = 2*(np.random.random(N)-0.5)*0.6
d1 = 2*(np.random.random(N)-0.5)
A = lambda sc,de : lambda d : sc*d*np.exp(-np.abs(d)/de) 
alpha = A(5,1)
sigma = 0.06
y = (np.random.rand(N)<phi(df/sigma-alpha(d1))).astype(int)

plt.plot(d1,alpha(d1),'rx',)
plt.plot(d1,df/sigma,'gx',)


