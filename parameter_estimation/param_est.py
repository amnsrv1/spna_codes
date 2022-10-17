
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
dm_modes = 40
sc_modes = 50



#PLEASE GIVE PULSAR NAME HERE

psrname = "J1909-3744"


#####Importing parfile and timfile#####


parfile = "../"+psrname+".NB.par"
timfile = "../"+psrname+".NB.tim"

psr = Pulsar(parfile, timfile,ephem='DE438')

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
efac = parameter.Uniform(1, 10)
equad = parameter.Uniform(-9,0)

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
#eq = white_signals.EquadNoise(log10_equad=equad, selection=selection_by_groups)
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


#Please enter final model obtained from previous steps


model = wn + arn + dmn + scn + tm 


# initialize PTA
pta = signal_base.PTA([model(psr)])



#USING PTMCMC (parameter estimation)

x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)
# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution

# where chains will be written to
outdir = 'output_param_est/'.format(str(psr.name))


# sampler object
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov,
                 outDir=outdir, 
                 resume=False)



# sampler for N steps
N = int(1e6)

# SCAM = Single Component Adaptive Metropolis
# AM = Adaptive Metropolis
# DE = Differential Evolution
## You can keep all these set at default values
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)


et = time.time()
ep_time = (et-st)
print('elapsed time in ptmcmc:', ep_time, 's')



