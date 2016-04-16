code_path = '/home/vincent/git/io-pitch-discrimination/code/python/'

import sys
sys.path.append('/home/vincent/git/independent-jobs/')
sys.path.append('/nfs/nhome/live/vincenta/git/independent-jobs')
sys.path.append(code_path)

from independent_jobs.aggregators.SingleResultAggregator import SingleResultAggregator

from parallel.parallel_lh import *

# Declaring grid values


a = np.log(500)
b = np.log(2000)
mu_p = 0.5*(a+b)
s_p = 1./12.*(b-a)**2

mu_g = mu_p #assumption that prior mean is well learned
n_grid_g = 10
n_grid_s = 10
n_grid_h = 10
n_samp = 4000
S_G = np.linspace(0.02*s_p,s_p,n_grid_g) # from 1/10 to twice the true variance
S_S = np.linspace((b-a)/100.,(b-a)/10.,n_grid_s) #
H = np.linspace(0.0,2.,n_grid_h)
extent = [S_G.min(),S_G.max(),S_S.min(),S_S.max()]

data_path = '/home/vincent/data/Itay_data/wideRange.mat'
loader = Dataloader(data_path)
th = 0.7
I_poor,I_good = loader.group_by_acc(th)

submit_type = 'slurm'
save_dir = expanduser("~")+'/pitch_shift_results2oct'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



#  Iterate over task and values
logger.info("Starting loop over job submission")

for I,group in zip([I_poor,I_good],['poor','good']):
    index = 0
    F1,F2,Y = loader.subject_data(I)
    F1 = F1.flatten()
    F2 = F2.flatten()
    Y = Y.flatten()

    for i_g,s_g in enumerate(S_G):
        for i_s,s_s in enumerate(S_S):
            for i_h,h in enumerate(H):


                #---------------------
                # setting model parameters
                mod_prm = {'mu_g':mu_g,'s_g':s_g,'h':h,'s_s':s_s,'n_samp':n_samp}
                data = {'F1':F1,'F2':F2,'Y':Y}
                #---------------------

                name = 'llh_'+group+'_'+str(index)
                duration_job_min=60*20
                engine = prepare_engine(submit_type=submit_type,
                                    duration_job_min=duration_job_min)

                aggregators = []

                logger.info("Submitting job")
                job = Sim_parallel_job(SingleResultAggregator(),
                                            mod_prm=mod_prm,
                                            data=data,
                                            name=name,
                                            save_dir=save_dir)

                aggregators.append(engine.submit_job(job))

                index+=1
                logger.info("Don't Wait for all call in engine")


# =================

