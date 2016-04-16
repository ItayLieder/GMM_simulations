code_path = '/home/vincent/git/io-pitch-discrimination/code/python/'

import sys
sys.path.append('/home/vincent/git/independent-jobs/')
sys.path.append('/nfs/nhome/live/vincenta/git/independent-jobs')
sys.path.append(code_path)

from model.simple_model import *

from independent_jobs.jobs.IndependentJob import IndependentJob
from independent_jobs.results.SingleResult import SingleResult
from independent_jobs.engines.BatchClusterParameters import BatchClusterParameters
from independent_jobs.engines.SlurmComputationEngine import SlurmComputationEngine
from independent_jobs.engines.SerialComputationEngine import SerialComputationEngine
from independent_jobs.tools.Log import logger, Log

import os
import time
import pickle

from os.path import expanduser

"""
Simulate and store
"""
def prepare_engine(submit_type='local',
                    duration_job_min=60*4):
    # ---------------------
    Log.set_loglevel(20)
    logger.info("Start")

    foldername = expanduser("~")+'/slurm_jobs'
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    logger.info("Setting engine folder to %s" % foldername)
    logger.info("Creating batch parameter instance")
    johns_slurm_hack = "#SBATCH --partition=intel-ivy,wrkstn,compute"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    batch_parameters = BatchClusterParameters(max_walltime=duration_job_min,
        foldername=foldername,
        job_name_base="sim_"+timestr+"_",
        parameter_prefix=johns_slurm_hack)

    if submit_type =='slurm':
        logger.info("Creating slurm engine instance")
        engine = SlurmComputationEngine(batch_parameters)
    elif submit_type == "local":
        logger.info("Creating serial engine instance")
        engine = SerialComputationEngine()
    # ---------------------

    return engine

class Sim_parallel_job(IndependentJob):
    def __init__(self, aggregator,
                 mod_prm=None,
                 data=None,
                 name=None,
                 save_dir=None):
        IndependentJob.__init__(self, aggregator)
        self.mod_prm = mod_prm
        self.data = data
        self.name = name
        self.save_dir=save_dir

    def compute(self, data,mod_prm):
        logger.info("computing")
#
        #------------------------------
        mu_g = mod_prm['mu_g']
        s_g = mod_prm['s_g']
        h = mod_prm['h']
        s_s = mod_prm['s_s']
        model = Model(mu_g,s_g,h,s_s)
        #------------------------------
        F1 = data['F1']
        F2 = data['F2']
        Y = data['Y']
        #------------------------------
        x = model.llh(F1,F2,Y,n_samp=n_samp)
        #------------------------------
#
        result = SingleResult([x])
        self.aggregator.submit_result(result)
        logger.info("done computing")
        mypath = self.save_dir+'/'+self.name+".p"
        logger.info("saving:"+mypath)

        d = {'data':self.data,'mod_prm':self.mod_prm,'llh':x}
        pickle.dump( d, open(mypath, "wb" ) )

