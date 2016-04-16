import numpy as np
from abc import abstractmethod

from lib.truncated_normal_ab_pdf import truncated_normal_ab_pdf
from lib.truncated_normal_ab_sample import truncated_normal_ab_sample
from lib.normal_01_cdf import normal_01_cdf

class Prior(object):

    @abstractmethod
    def stats(self, F):
        raise NotImplementedError()

    @abstractmethod
    def log_pdf(self, f):
        raise NotImplementedError()


class Likelihood(object):

    @abstractmethod
    def stats(self, F):
        raise NotImplementedError()


    @abstractmethod
    def log_pdf(self, f):
        raise NotImplementedError()

    @abstractmethod
    def sample(self, n=1):
        raise NotImplementedError()


class Distribution(object):

    @abstractmethod
    def sample(self,N):
        raise NotImplementedError()

    @abstractmethod
    def log_pdf(self,x):
        raise NotImplementedError()

    def pdf(self,x):
        return np.exp(self.log_pdf(x))

    @abstractmethod
    def posterior(self,lh):
        """
        Posterior under Gaussian likelihood
        """
        raise NotImplementedError()


class Gaussian(Distribution):

    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self,x):
        return -0.5*np.log(2.*np.pi)- np.log(self.sigma) - 0.5/self.sigma**2*(x-self.mu)**2

    def sample(self,N):
        return self.mu + self.sigma*np.random.randn(N)

    def posterior(self,lh):
        assert isinstance(lh,Gaussian)
        _,g = GaussianProduct(self,lh)
        return g


class UnnormalizedGaussian(Distribution):
    ### A gaussian with scaling such that p(x=mu)=1

    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma

    def log_pdf(self,x):
        return - 0.5/self.sigma**2*(x-self.mu)**2

    def sample(self,N):
        # sampling unaffected by truncation
        return self.mu + self.sigma*np.random.randn(N)

    def posterior(self,lh):
        assert isinstance(lh,Gaussian)
        _,g = GaussianProduct(self,lh)
        return g

class TruncatedGaussian(Distribution):

    def __init__(self,mu,sigma,a,b):
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b

    def pdf(self,x):
        return truncated_normal_ab_pdf(x,self.mu,self.sigma,self.a,self.b)*((x>self.a)&(x<self.b))

    def log_pdf(self,x):
        return np.log(self.pdf(x))

    def sample(self,N):
        s = np.zeros((N,))
        seed = 123456789;
        for i in range(N):
            [x,seed] = truncated_normal_ab_sample ( self.mu, self.sigma, self.a, self.b,seed);
            s[i] = x
        return s


    def posterior(self,lh):
        raise NotImplementedError()



class Uniform(Distribution):

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def log_pdf(self,x):
        return -np.log(self.b-self.a)*((x>=self.a)&(x<=self.b))

    def sample(self,N):
        return self.a+np.random.rand(N)*(self.b-self.a)

    def posterior(self,lh):
        assert isinstance(lh,Gaussian)
        return Gaussian(lh.mu,lh.sigma)


class RealUniform(Distribution):

    def __init__(self,a,b):
        self.a = a
        self.b = b

    def log_pdf(self,x):
        y = -np.Inf*np.ones(x.shape)
        y[(x>=self.a)&(x<=self.b)] = -np.log(self.b-self.a)
        return y

    def pdf(self,x):
        y = np.zeros(x.shape)
        y[(x>=self.a)&(x<=self.b)] = 1./(self.b-self.a)
        return y

    def sample(self,N):
        return self.a+np.random.rand(N)*(self.b-self.a)

    def posterior(self,lh):
        assert isinstance(lh,Gaussian)
        return TruncatedGaussian(lh.mu,lh.sigma,self.a,self.b)


class Mixture(Distribution):

    def __init__(self,cpts,w):
        assert isinstance(cpts,list)
        for item in cpts:
            assert isinstance(item,Distribution)
        assert len(w)==len(cpts)
        self.cpts = cpts
        self.w = w

    def log_pdf(self,x):
        return np.log(self.pdf(x))

    def pdf(self,x):
        y = np.zeros(x.shape)
        for cpt,w in zip(self.cpts,self.w):
            y+=np.exp(cpt.log_pdf(x))*w
        return y

    def sample(self,N):
        inds = sample_discrete(self.w,N)
        Ns = [np.sum(inds==k) for k in range(len(self.w))]
        samples = []
        for i,n in enumerate(Ns):
            samples.append( self.cpts[i].sample(n) )
        return np.hstack(samples)

    def posterior(self,lh):
        assert isinstance(lh,Gaussian)

        lks =[]
        new_cpts = []
        for cpt in self.cpts:
            lk,d = GaussianProduct(cpt,lh)
            lks.append(lk)
            new_cpts.append(d)
        eps = 1e-30
        lw = np.array(lks)+ np.log(np.array(self.w) + eps)
        new_w = np.exp(lw - np.max(lw))
        new_w = new_w/np.sum(new_w)

        return Mixture(new_cpts,new_w)


def GaussianProduct(p,lh):
    """
    Output is a gaussian with a scaling factor
    :param d: the prior
    :param g: the likelihood
    :return: log scaling and gaussian
    """
    assert isinstance(p,Distribution)
    assert isinstance(lh,Gaussian)

    if isinstance(p,Gaussian):
        s_po = (1./p.sigma**2+ 1./lh.sigma**2)**(-0.5)
        mu_po = s_po**2*(p.mu/p.sigma**2 + lh.mu/lh.sigma**2)
        lk = -0.5*np.log(2.*np.pi) - np.log(p.sigma*lh.sigma/s_po)\
          -0.5*(p.mu**2/p.sigma**2 + lh.mu**2/lh.sigma**2 - mu_po**2/s_po**2 )
        return lk, Gaussian(mu_po,s_po)

    elif isinstance(p,Uniform):
        return -np.log(p.b-p.a), Gaussian(lh.mu, lh.sigma)

    elif isinstance(p,RealUniform):
        alpha = ( p.a - lh.mu ) / lh.sigma
        beta = ( p.b - lh.mu ) / lh.sigma
        alpha_cdf = normal_01_cdf ( alpha )
        beta_cdf = normal_01_cdf ( beta )

        lk = np.log((beta_cdf-alpha_cdf)*lh.sigma)

        return lk, TruncatedGaussian(lh.mu, lh.sigma,p.a,p.b)

    elif isinstance(p,UnnormalizedGaussian):
        s_po = (1./p.sigma**2+ 1./lh.sigma**2)**(-0.5)
        mu_po = s_po**2*(p.mu/p.sigma**2 + lh.mu/lh.sigma**2)
        lk = - np.log(lh.sigma/s_po)\
          -0.5*(p.mu**2/p.sigma**2 + lh.mu**2/lh.sigma**2 - mu_po**2/s_po**2 )
        return lk, Gaussian(mu_po,s_po)

    else:
        raise NotImplementedError()



#######################################################

def sample_discrete(w,N):
    return np.array(w).cumsum().searchsorted(np.random.rand(N))

