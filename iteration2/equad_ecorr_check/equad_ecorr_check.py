
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

psrname = ""

#####Importing parfile and timfile#####



parfile = "../"+psrname+".NB.par"
timfile = "../"+psrname+".NB.tim"

psr = Pulsar(parfile, timfile,ephem='DE440')



tmin = psr.toas.min()
tmax = psr.toas.max()
Tspan = np.max(tmax) - np.min(tmin)
Tspan_chrom = 4 * Tspan
Tspan_years = Tspan/ 365.25 / 24 / 60 / 60
print(Tspan_years)  # time span of data in years



def by_groups(flags):
    """Selection function to split by -group flag"""
    flagvals = np.unique(flags["group"])
    return {val: flags["group"] == val for val in flagvals}

selection_by_groups = Selection(by_groups)


#Setting up Noise priors

#White noise
efac = parameter.Uniform(0.1, 8)
equad = parameter.Uniform(-8,-2)
ecorr = parameter.Uniform(-8, -2)


#White noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection_by_groups)
wn = white_signals.MeasurementNoise(efac=efac,log10_t2equad = equad, selection=selection_by_groups) 
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection_by_groups)

# timing model
tm = gp_signals.TimingModel(use_svd=True, normed=True, coefficients=False)

#LET US DEFINE 4 DIFFERENT MODEL COMBINATIONS TO COMPARE THEIR EVIDENCES

model1 = ef + tm
model2 = wn + tm
model3 = ef + ec +  tm
model4 = wn + ec + tm


# initialize PTA

pta1 = signal_base.PTA([model1(psr)])
pta2 = signal_base.PTA([model2(psr)])
pta3 = signal_base.PTA([model3(psr)])
pta4 = signal_base.PTA([model4(psr)])


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



#creating a txt file for saving output
f= open("equad_ecorr_check_"+psrname+".txt","w+")



#Dynesty sampler run and plotting
Dres1 = dynesty_sample(pta1)
print("For model 1, log evidence = {} ± {}".format(Dres1.logz[-1], Dres1.logzerr[-1]), file = f)
weights1 = np.exp(Dres1['logwt'] - Dres1['logz'][-1])
samples1 = resample_equal(Dres1.samples, weights1)
fig1 = corner.corner(samples1,labels=list(pta1.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model1.pdf')




Dres2 = dynesty_sample(pta2)
print("For model 2, log evidence = {} ± {}".format(Dres2.logz[-1], Dres2.logzerr[-1]), file = f)
weights2 = np.exp(Dres2['logwt'] - Dres2['logz'][-1])
samples2 = resample_equal(Dres2.samples, weights2)
fig2 = corner.corner(samples2,labels=list(pta2.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model2.pdf')




Dres3 = dynesty_sample(pta3)
print("For model 3, log evidence = {} ± {}".format(Dres3.logz[-1], Dres3.logzerr[-1]), file = f)
weights3 = np.exp(Dres3['logwt'] - Dres3['logz'][-1])
samples3 = resample_equal(Dres3.samples, weights3)
fig3 = corner.corner(samples3,labels=list(pta3.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model3.pdf')




Dres4 = dynesty_sample(pta4)
print("For model 4, log evidence = {} ± {}".format(Dres4.logz[-1], Dres4.logzerr[-1]), file = f) 
weights4 = np.exp(Dres4['logwt'] - Dres4['logz'][-1])
samples4 = resample_equal(Dres4.samples, weights4)
fig4 = corner.corner(samples4,labels=list(pta4.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model4.pdf')



et = time.time()
ep_time = (et-st)
print('elapsed time:', ep_time, 's')  


f.close()
