import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from distributions import *
from abc import abstractmethod


class AbstractModel(object):
    """
    Abstract class for all simple models
    """
    @abstractmethod
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def bern(self,f1,f2,n_samp=100000):
        raise NotImplementedError()

    def llh(self,F1,F2,Y,n_samp=5000,skip=None):
        """probability f1 higher"""
        assert len(F1.shape)==1, 'input must be 1d ndarray'
        assert len(F1.shape)==1, 'input must be 1d ndarray'
        assert len(Y.shape)==1, 'input must be 1d ndarray'
        assert (F1.shape == F2.shape)&(F1.shape == Y.shape)
        eps = 1.e-5
        B = self.bern(F1,F2,n_samp=n_samp)
        B[B<eps]=eps
        B[B>1-eps]=1-eps
        if skip==None:
            skip = np.ones(Y.shape).astype(bool)
        else:
            assert isinstance(skip,np.ndarray)
        llh = np.sum( (Y*np.log(B) + (1-Y)*np.log(1-B))[skip]  )
        return llh

    def grid_evaluation(self,F1,F2,n_samp=5000):
        """Evaluation of %f1 higher on a 2d grid of f1/f2 values"""
        assert len(F1.shape)==1, 'input must be 1d ndarray'
        assert len(F2.shape)==1, 'input must be 1d ndarray'
        assert (F1.shape == F2.shape)
        n1 = len(F1)
        n2 = len(F2)
        f1,f2 = np.meshgrid(F1,F2)
        B = self.bern(f1.flatten(),f2.flatten(),n_samp=n_samp).reshape(n1,n2)
        return B


class Model(AbstractModel):
    """
    A model of tone discrimination where
    - the value of the first tone is inferred from noisy observation
    - the value of the second tone is noiseless
    - prior is unigauss
    """

    def __init__(self, mu_g,s_g,h,s_s):
        """
        Constructor
        :param mu_g: mean of gaussian part of unigauss
        :param s_g: std of gaussian part of unigauss
        :param h: weight of flat prior in unigauss mixture assuming unnormalized gaussian p(x) 1/Z*( h + exp((x-mu)/2/s^2) )
        :param s_s: std of likelihood
        """
        self.mu_g = mu_g
        self.s_g = s_g
        self.s_s = s_s
        self.h = h

    def bern(self,f1,f2,n_samp=100000):
        """probability f1 higher"""
        assert len(f1.shape)==1, 'input must be 1d ndarray'
        assert len(f2.shape)==1, 'input must be 1d ndarray'
        assert f1.shape == f2.shape

        n_trial = len(f1)
        # Takes 1d array as input
        # sensory noise
        f1_ = np.tile(f1,(n_samp,1)) + self.s_s*np.random.randn(n_samp,n_trial)
        s = (1./self.s_g**2 + 1./self.s_s**2)**(-1./2.)
        mu = s**2*(self.mu_g/self.s_g**2 + f1_/self.s_s**2)
        if self.h == 0:
            p = norm.cdf(f2,mu,s)
        else:
            # posterior inference
            #k = 1./np.sqrt(2.*np.pi)*s/(self.s_g*self.s_s)\
            #    *np.exp(-0.5*(self.mu_g**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2))
            k_g = s/self.s_s\
                *np.exp(-0.5*(self.mu_g**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2)) # for unnormalized gaussian prior
            pi_u = self.h/(self.h+k_g)
            pi_g = 1.-pi_u
            # decision (probability f1 lower)
            p = pi_g*norm.cdf(f2,mu,s) + pi_u*norm.cdf(f2,f1_,self.s_s)

        return 1.-np.mean(p,axis=0) # average over samples

    def f1_post_mean(self,f1,n_samp=5000):
        """posterior mean on f1"""
        assert len(f1.shape)==1, 'input must be 1d ndarray'
        n_trial = len(f1)
        # Takes 1d array as input
        # sensory noise
        f1_ = np.tile(f1,(n_samp,1)) + self.s_s*np.random.randn(n_samp,n_trial)
        s = (1./self.s_g**2 + 1./self.s_s**2)**(-1./2.)
        mu = s**2*(self.mu_g/self.s_g**2 + f1_/self.s_s**2)
        if self.h == 0:
            return np.mean(mu,axis=0)
        else:
            # posterior inference
            #k = 1./np.sqrt(2.*np.pi)*s/(self.s_g*self.s_s)\
            #    *np.exp(-0.5*(self.mu_g**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2))
            k_g = s/self.s_s\
                *np.exp(-0.5*(self.mu_g**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2)) # for unnormalized gaussian prior
            pi_u = self.h/(self.h+k_g)
            pi_g = 1.-pi_u

            return np.mean(pi_u*f1_+pi_g*mu,axis=0)


class NoPriorModel(AbstractModel):
    def __init__(self, s_s):
        """
        Constructor
        :param s_s: std of likelihood
        """
        self.s_s = s_s

    def bern(self,f1,f2,n_samp=100000):
        """probability f1 higher"""
        assert len(f1.shape)==1, 'input must be 1d ndarray'
        assert len(f2.shape)==1, 'input must be 1d ndarray'
        assert f1.shape == f2.shape

        n_trial = len(f1)
        # Takes 1d array as input
        # sensory noise
        f1_ = np.tile(f1,(n_samp,1)) + self.s_s*np.random.randn(n_samp,n_trial)
        s = self.s_s
        mu = f1_
        p = norm.cdf(f2,mu,s)
        return 1.-np.mean(p,axis=0) # average over samples


class LocalModel(AbstractModel):
    """
    Same as recency model except that gaussian prior mean is set to average of previous tones
    """

    def bern(self,f1,f2,n_samp=100000):
        """probability f1 higher"""
        assert len(f1.shape)==1, 'input must be 1d ndarray'
        assert len(f2.shape)==1, 'input must be 1d ndarray'
        assert f1.shape == f2.shape

        n_trial = len(f1)
        # Takes 1d array as input
        # sensory noise
        f1_ = np.tile(f1,(n_samp,1)) + self.s_s*np.random.randn(n_samp,n_trial)
        s = (1./self.s_g**2 + 1./self.s_s**2)**(-1./2.)

        mu_prev = np.zeros(f1.shape)
        mu_prev[:-1] = 0.5*(f1+f2)[1:]
        mu_prev[-1] = self.mu_g


        mu = s**2*(mu_prev/self.s_g**2 + f1_/self.s_s**2)
        if self.h == 0:
            p = norm.cdf(f2,mu,s)
        else:
            # posterior inference
            #k = 1./np.sqrt(2.*np.pi)*s/(self.s_g*self.s_s)\
            #    *np.exp(-0.5*(self.mu_g**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2))
            k_g = s/self.s_s\
                *np.exp(-0.5*(mu_prev**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2)) # for unnormalized gaussian prior
            pi_u = self.h/(self.h+k_g)
            pi_g = 1.-pi_u
            # decision (probability f1 lower)
            p = pi_g*norm.cdf(f2,mu,s) + pi_u*norm.cdf(f2,f1_,self.s_s)

        return 1.-np.mean(p,axis=0) # average over samples


class LocalGlobalModel(AbstractModel):
    """
    Same as recency model except that gaussian prior mean is set somewhere between
    - average of previous tones
    - long term prior mean
    """

    def __init__(self, mu_g,s_g,h,s_s,alpha=0.):
        """
        Constructor
        :param mu_g: mean of gaussian part of unigauss
        :param s_g: std of gaussian part of unigauss
        :param h: weight of flat prior in unigauss mixture assuming unnormalized gaussian p(x) 1/Z*( h + exp((x-mu)/2/s^2) )
        :param s_s: std of likelihood
        :param alpha: interpolation factor 1=local,0=global
        """
        self.mu_g = mu_g
        self.s_g = s_g
        self.s_s = s_s
        self.h = h
        self.alpha = alpha


    def bern(self,f1,f2,n_samp=100000):
        """probability f1 higher"""
        assert len(f1.shape)==1, 'input must be 1d ndarray'
        assert len(f2.shape)==1, 'input must be 1d ndarray'
        assert f1.shape == f2.shape

        n_trial = len(f1)
        # Takes 1d array as input
        # sensory noise
        f1_ = np.tile(f1,(n_samp,1)) + self.s_s*np.random.randn(n_samp,n_trial)
        s = (1./self.s_g**2 + 1./self.s_s**2)**(-1./2.)

        mu_prev = np.zeros(f1.shape)
        mu_prev[1:] = 0.5*(f1+f2)[:-1]
        mu_prev[0] = self.mu_g

        mu_eff = mu_prev*self.alpha + (1.-self.alpha)*self.mu_g

        mu = s**2*(mu_eff/self.s_g**2 + f1_/self.s_s**2)
        if self.h == 0:
            p = norm.cdf(f2,mu,s)
        else:
            # posterior inference
            #k = 1./np.sqrt(2.*np.pi)*s/(self.s_g*self.s_s)\
            #    *np.exp(-0.5*(self.mu_g**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2))
            k_g = s/self.s_s\
                *np.exp(-0.5*(mu_eff**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2)) # for unnormalized gaussian prior
            pi_u = self.h/(self.h+k_g)
            pi_g = 1.-pi_u
            # decision (probability f1 lower)
            p = pi_g*norm.cdf(f2,mu,s) + pi_u*norm.cdf(f2,f1_,self.s_s)

        return 1.-np.mean(p,axis=0) # average over samples


class RecencyModel(AbstractModel):

    def __init__(self, s_g,s_s,alpha=0.,h=0.,ratio_s_g=1):
        """
        Constructor
        :param s_g: std of gaussian part of unigauss
        :param h: weight of flat prior in unigauss mixture assuming unnormalized gaussians sharing the same variance
        :param s_s: std of likelihood
        :param alpha: ratio between gaussians
        """
        self.s_g = s_g
        self.s_s = s_s
        self.h=h
        self.alpha = alpha
        self.ratio_s_g = ratio_s_g


    def bern(self,f1,f2,n_samp=100000):
        """probability f1 higher"""
        assert len(f1.shape)==1, 'input must be 1d ndarray'
        assert len(f2.shape)==1, 'input must be 1d ndarray'
        assert f1.shape == f2.shape

        n_trial = len(f1)
        # Takes 1d array as input
        # sensory noise
        f1_ = np.tile(f1,(n_samp,1)) + self.s_s*np.random.randn(n_samp,n_trial)
        s_g2 = self.ratio_s_g*self.s_g
        s_1 = (1./self.s_g**2 + 1./self.s_s**2)**(-1./2.)
        s_2 = (1./s_g2**2 + 1./self.s_s**2)**(-1./2.)

        mu_prev1 = np.zeros(f1.shape)
        mu_prev2= np.zeros(f1.shape)
        mu_prev1[1:] = 0.5*(f1+f2)[:-1]
        mu_prev1[0] = (f1+f2)[0] # arbitrary
        mu_prev2[2:] = 0.5*(f1+f2)[:-2]
        mu_prev2[0:1] = (f1+f2)[0] # arbitrary


        mu1 = s_1**2*(mu_prev1/self.s_g**2 + f1_/self.s_s**2)
        mu2 = s_2**2*(mu_prev2/s_g2**2 + f1_/self.s_s**2)

        # posterior inference
        #k = 1./np.sqrt(2.*np.pi)*s/(self.s_g*self.s_s)\
        #    *np.exp(-0.5*(self.mu_g**2/self.s_g**2+f1_**2/self.s_s**2-mu**2/s**2))
        k_g1 = self.alpha*s_1/self.s_s\
            *np.exp(-0.5*(mu_prev1**2/self.s_g**2+f1_**2/self.s_s**2-mu1**2/s_1**2)) # for unnormalized gaussian prior
        k_g2 = (1-self.alpha)*s_2/self.s_s\
            *np.exp(-0.5*(mu_prev2**2/s_g2**2+f1_**2/self.s_s**2-mu2**2/s_2**2)) # for unnormalized gaussian prior
        k_u = self.h

        p_g1=k_g1/(k_g2+k_g1+k_u)
        p_g2=k_g2/(k_g2+k_g1+k_u)
        p_u=k_u/(k_g2+k_g1+k_u)

        # decision (probability f1 lower)
        p = p_g1*norm.cdf(f2,mu1,s_1) +\
            p_g2*norm.cdf(f2,mu2,s_2) +\
            p_u*norm.cdf(f2,f1_,self.s_s)

        return 1.-np.mean(p,axis=0) # average over samples



if __name__ == '__main__':
    a = np.log(500)
    b = np.log(2000)
    mu_p = 0.5*(a+b)
    s_p = 1./12.*(b-a)**2
    #------------------
    n_grid = 40
    n_samp = 5000
    F = np.linspace(a,b,n_grid)
    B = np.zeros((n_grid,n_grid))
    extent=[F.min(),F.max(),F.min(),F.max()]
    mu_g = mu_p
    s_g = 0.5*s_p
    h = 0.1
    s_s = 0.1


    model = Model(mu_g,s_g,h,s_s)
    B = model.grid_evaluation(F,F)
    plt.imshow(B,extent=extent,
               interpolation='nearest',
              origin='lower',alpha=0.9)
    plt.xlabel('$f_1$')
    plt.ylabel('$f_2$')
    plt.contour(B,extent=extent,c='k')
    plt.show()