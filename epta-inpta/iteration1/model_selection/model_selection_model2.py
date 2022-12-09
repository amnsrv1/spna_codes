import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg as sl
import math
import os
from multiprocess import Pool
from dynesty.utils import resample_equal

import nestle
import dynesty
from dynesty import plotting as dyplot
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
import enterprise.signals.selections as selections

from enterprise import constants as const
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise.signals import (gp_signals, parameter, selections, utils,
                                white_signals)
from enterprise_extensions import deterministic as ee_deterministic
from enterprise_extensions import chromatic as chrom
from enterprise_extensions import dropout as drop
from enterprise_extensions import gp_kernels
from enterprise_extensions import model_orfs
from enterprise_extensions.blocks import (dm_noise_block,red_noise_block,chromatic_noise_block)
import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


import time
st = time.time()


#PLEASE GIVE PULSAR NAME HERE

psrname = "J1909-3744"

#creating a txt file for saving output
f= open("model_selection_model2"+psrname+".txt","w+")



#####Importing parfile and timfile#####

parfile = "../"+psrname+".par"
timfile = "../"+psrname+"_all.tim"

psr = Pulsar(parfile, timfile,ephem='DE440')


tmin = psr.toas.min()
tmax = psr.toas.max()
Tspan = np.max(tmax) - np.min(tmin)
Tspan_years = Tspan/ 365.25 / 24 / 60 / 60
print(Tspan_years)  # time span of data in years


#No. of modes for rednoises

arn_modes = 30
dm_modes =  100


#selection type for white noise
def by_groups(flags):
    """Selection function to split by -group flag"""
    flagvals = np.unique(flags["group"])
    return {val: flags["group"] == val for val in flagvals}
selection_by_groups = Selection(by_groups)


#Setting up Noise priors

#White noise
efac = parameter.Uniform(0.1, 8)
equad = parameter.Uniform(-8,-2)


#White noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection_by_groups)
wn = white_signals.MeasurementNoise(efac=efac,log10_t2equad = equad, selection=selection_by_groups)


arn = red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan,
                    components=arn_modes, gamma_val=None, coefficients=False,
                    select=None, modes=None, wgts=None, combine=True,
                    break_flat=False, break_flat_fq=None,
                    logmin=None, logmax=None, dropout=False, k_threshold=0.5)



dmn = dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', dt=15, df=200,
                   Tspan=Tspan, components=dm_modes,
                   gamma_val=None, coefficients=False)



# timing model
tm = gp_signals.TimingModel(use_svd=True, normed=True, coefficients=False)

#LET US DEFINE 4 DIFFERENT MODEL COMBINATIONS TO COMPARE THEIR EVIDENCES

model2 = wn + dmn + tm


# initialize PTA

pta2 = signal_base.PTA([model2(psr)])


### PRIOR TRANSFORM, DYNESTY


def prior_transform_fn(pta):
    mins = np.array([param.prior.func_kwargs['pmin'] for param in pta.params])
    maxs = np.array([param.prior.func_kwargs['pmax'] for param in pta.params])
    spans = maxs-mins
    
    def prior_transform(cube):
        return spans*cube + mins
    
    return prior_transform

def dynesty_sample(pta):
    prior_transform = prior_transform_fn(pta)
    with  Pool() as pool:
        sampler = dynesty.NestedSampler(pta.get_lnlikelihood, prior_transform, len(pta.params),bound='multi',pool=pool,queue_size=os.cpu_count(), bootstrap = 0)
        sampler.run_nested(dlogz = 0.1, print_progress=False )
    res = sampler.results
    return res


#Dynesty sampler run and plotting

Dres2 = dynesty_sample(pta2)
print("For model 2, log evidence = {} ± {}".format(Dres2.logz[-1], Dres2.logzerr[-1]))
print("For model 2, log evidence = {} ± {}".format(Dres2.logz[-1], Dres2.logzerr[-1]), file = f)
weights2 = np.exp(Dres2['logwt'] - Dres2['logz'][-1])
samples2 = resample_equal(Dres2.samples, weights2)
fig2 = corner.corner(samples2,labels=list(pta2.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model2.pdf')



et = time.time()
ep_time = (et-st)
print('elapsed time:', ep_time, 's')  


f.close()

