
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg as sl

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


#No. of modes for rednoises

arn_modes = 50
dm_modes =  50
sc_modes =  50


##### Set up signals #####


parfile = "../J1909-3744.par"
timfile = "../J1909-3744.NB.tim"


psr = Pulsar(parfile, timfile)


tmin = psr.toas.min()
tmax = psr.toas.max()
Tspan = np.max(tmax) - np.min(tmin)
print(Tspan / 365.25 / 24 / 60 / 60)  # time span of data in years


def by_groups(flags):
    """Selection function to split by -group flag"""
    flagvals = np.unique(flags["group"])
    return {val: flags["group"] == val for val in flagvals}

selection_by_groups = Selection(by_groups)


#Setting up Noise priors

#White noise
efac = parameter.Uniform(0.1, 5)
equad = parameter.Uniform(-9,-5)

#AchromaticRedNoise
log10_A = parameter.Uniform(-18, -10)
gamma = parameter.Uniform(0, 7)

#ChromaticRedNoise
chr_idx = parameter.Uniform(0, 5)
chr_log10_A = parameter.Uniform(-18, -10)
chr_gamma = parameter.Uniform(0, 7)

#DMNoise
dm_log10_A = parameter.Uniform(-18, -10)
dm_gamma = parameter.Uniform(0, 7)

#ScatteringNoise
sc_log10_A = parameter.Uniform(-18, -10)
sc_gamma = parameter.Uniform(0, 7)


#White noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection_by_groups)
wn = white_signals.MeasurementNoise(efac=efac,log10_t2equad = equad, selection=selection_by_groups) 



arn = red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan,
                    components=arn_modes, gamma_val=None, coefficients=False,
                    select=None, modes=None, wgts=None, combine=True,
                    break_flat=False, break_flat_fq=None,
                    logmin=None, logmax=None, dropout=False, k_threshold=0.5)



scn = chromatic_noise_block(gp_kernel='diag', psd='powerlaw',
                          nondiag_kernel='periodic',
                          prior='log-uniform', dt=15, df=200,
                          idx=4, include_quadratic=False,
                          Tspan=Tspan, name='scat', components=sc_modes,
                          coefficients=False)


dmn = dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', dt=15, df=200,
                   Tspan=Tspan, components=dm_modes,
                   gamma_val=None, coefficients=False)



# timing model
tm = gp_signals.TimingModel(use_svd=True, normed=True, coefficients=False)

#LET US DEFINE 4 DIFFERENT MODEL COMBINATIONS TO COMPARE THEIR EVIDENCES

model1 = wn + tm
model2 = wn + arn + tm
model3 = wn + arn + dmn +  tm
model4 = wn + arn + dmn + scn + tm

# initialize PTA

pta1 = signal_base.PTA([model1(psr)])
pta2 = signal_base.PTA([model2(psr)])
pta3 = signal_base.PTA([model3(psr)])
pta4 = signal_base.PTA([model4(psr)])



### PRIOR TRANSFORM, NESTLE


def prior_transform_fn(pta):
    mins = np.array([param.prior.func_kwargs['pmin'] for param in pta.params])
    maxs = np.array([param.prior.func_kwargs['pmax'] for param in pta.params])
    spans = maxs-mins
    
    def prior_transform(cube):
        return spans*cube + mins
    
    return prior_transform



def nestle_sample(pta):
    prior_transform = prior_transform_fn(pta)
    res = nestle.sample(pta.get_lnlikelihood, prior_transform, len(pta.params), 
                        method='multi', npoints=1000,dlogz = 0.1,callback=nestle.print_progress)
    return res


#Run the sampler for NESTLE for different models

Nres1 = nestle_sample(pta1)
print('for model 1 evidence is ',Nres1.logz, '+/-',Nres1.logzerr)
Nres2 = nestle_sample(pta2)
print('for model 1 evidence is ',Nres2.logz, '+/-',Nres2.logzerr)
Nres3 = nestle_sample(pta3)
print('for model 3 evidence is ',Nres3.logz, '+/-',Nres3.logzerr)
Nres4 = nestle_sample(pta4)
print('for model 4 evidence is ',Nres4.logz, '+/-',Nres4.logzerr)




samples_nestle1  = nestle.resample_equal(Nres1.samples, Nres1.weights)
fig1 = corner.corner(samples_nestle1,labels=list(pta1.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=False)
plt.suptitle("J1909-3744", fontsize=16)
plt.savefig('J1909_model1.pdf')



samples_nestle2  = nestle.resample_equal(Nres2.samples, Nres2.weights)
fig2 = corner.corner(samples_nestle2,labels=list(pta2.param_names), label_kwargs={"fontsize": 7},
                      quantiles=(0.16, 0.5, 0.84),show_titles=False)
plt.suptitle("J1909-3744", fontsize=16)
plt.savefig('J1909_model2.pdf')



samples_nestle3 = nestle.resample_equal(Nres3.samples, Nres3.weights)
fig3 = corner.corner(samples_nestle3,labels=list(pta3.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=False)
plt.suptitle("J1909-3744", fontsize=16)
plt.savefig('J1909_model3.pdf')



samples_nestle4  = nestle.resample_equal(Nres4.samples, Nres4.weights)
fig4 = corner.corner(samples_nestle4,labels=list(pta4.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=False)
plt.suptitle("J1909-3744", fontsize=16)
plt.savefig('J1909_model4.pdf')



et = time.time()
ep_time = (et-st)
print('elapsed time:', ep_time, 's')  
