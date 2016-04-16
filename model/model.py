from distributions import *

class Model(object):

    def __init__(self, lh1,lh2, prior,Nn=100.,Ni=100.):

        assert isinstance(prior,Distribution)
        assert isinstance(lh1,Distribution)
        assert isinstance(lh2,Distribution)

        self.lh1 = lh1
        self.lh2 = lh2
        self.prior = prior
        self.Nn = Nn
        self.Ni = Ni

    def approximate(self, f1,f2):

        ps = np.zeros((self.Nn,)) # conditioned on noise

        # sample noise

        f1n = f1 + self.lh1.sample(self.Nn)
        f2n = f2 + self.lh2.sample(self.Nn)

        # sample posterior
        i = 0

        s_n1 = self.lh1.sigma
        s_n2 = self.lh2.sigma

        for f1n_,f2n_ in zip(f1n,f2n):

            lh1 = Gaussian(f1n_,self.lh1.sigma)
            lh2 = Gaussian(f2n_,self.lh2.sigma)

            f1p = self.prior.posterior(lh1).sample(self.Ni)
            f2p = self.prior.posterior(lh2).sample(self.Ni)

            ps[i] = np.mean(f1p>f2p)
            i+=1

        pd = np.mean(ps)

        return pd


