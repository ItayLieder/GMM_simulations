import numpy as np
from scipy.io import loadmat


class Dataloader(object):
    """
    Class to load subject data, assuming it contains s1,s2,acc,resp
    Loader is tied to a mat file
    """
    def __init__(self, fname):
        self.fname = fname

    def subject_data(self,i_sub,flat=False):
        """
        Loading subject data
        :param i_sub:
        :return: stimuli and response
        """
        data = loadmat(self.fname)
        F1 = np.log(np.array(data['s1'][i_sub,:]))
        F2 = np.log(np.array(data['s2'][i_sub,:]))
        Y = data['resp'][i_sub,:]
        if flat:
            return F1.flatten(),F2.flatten(),Y.flatten()
        else:
            return F1,F2,Y

    def group_by_acc(self,th):
        """
        Group subjects by accuracy
        :param th:
        :return: indices of subjects whose accuracy is below (resp above) the threshold
        """
        data = loadmat(self.fname)
        acc = np.mean(data['acc'],axis=1)
        order = np.argsort(acc)
        acc= acc[order]
        I_good = order[acc>th]
        I_poor = order[acc<=th]
        return I_poor, I_good

    def group_by_acc_interval(self, th_down=0.,th_up=1.):
        """
        Select subjects by accuracy range
        :param th_down: accuracy threshold below
        :param th_up: accuracy threshold above
        :return: indices of subjects
        """
        data = loadmat(self.fname)
        acc = np.mean(data['acc'],axis=1)
        order = np.argsort(acc)
        acc= acc[order]
        return order[(th_down<acc)&(acc<th_up)]


if __name__ == '__main__':
    fname = '/home/vincent/data/Itay_data/wideRange.mat'
    loader = Dataloader(fname)
    F1,F2,Y = loader.subject_data(range(3))