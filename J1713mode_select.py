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


parfile = "J1713+0747.par"
timfile = "J1713+0747.NB.tim"

psr = Pulsar(parfile, timfile)

tmin = psr.toas.min()
tmax = psr.toas.max()
Tspan = np.max(tmax) - np.min(tmin)
print(Tspan / 365.25 / 24 / 60 / 60)  # time span of data in years


# In[198]:


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


# In[ ]:


#White noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection_by_groups)
#eq = white_signals.EquadNoise(log10_equad=equad, selection=selection_by_groups)
wn = white_signals.MeasurementNoise(efac=efac,log10_t2equad = equad, selection=selection_by_groups) 


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

arn = [10,20,30,40,50]
dm = [30,40,50,60,70]
sc = [30,40,50,60,80,100]
crn = 100

for i in dm:
    for j in sc:
        dmn = dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', dt=15, df=200,
                   Tspan=Tspan, components=i,
                   gamma_val=None, coefficients=False)
        scn = chromatic_noise_block(gp_kernel='diag', psd='powerlaw',
                          nondiag_kernel='periodic',
                          prior='log-uniform', dt=15, df=200,
                          idx=4, include_quadratic=False,
                          Tspan=Tspan, name='scat', components=j,
                          coefficients=False)

	 # timing model
        tm = gp_signals.TimingModel(use_svd =True)

        model1 = ef + dmn + scn + tm
        
        pta1 = signal_base.PTA([model1(psr)])
        
        Nres1 = nestle_sample(pta1)
        
        print('For model 1 with modes',i,j,', the log evidence is ',Nres1.logz, '+/-',Nres1.logzerr)
'''
        
for k in scn:
	arn = red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan,
            components=50, gamma_val=None, coefficients=False,
                    select=None, modes=None, wgts=None, combine=True,
                    break_flat=False, break_flat_fq=None,
                    logmin=None, logmax=None, dropout=False, k_threshold=0.5)

	dmn = dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', dt=15, df=200,
                   Tspan=Tspan, components=40,
                   gamma_val=None, coefficients=False)

	scn = chromatic_noise_block(gp_kernel='diag', psd='powerlaw',
                          nondiag_kernel='periodic',
                          prior='log-uniform', dt=15, df=200,
                          idx=4, include_quadratic=False,
                          Tspan=Tspan, name='scat', components=k,
                          coefficients=False)
  
        # timing model
	tm = gp_signals.TimingModel(use_svd =True)
	model1 = ef + arn + dmn + tm

	pta1 = signal_base.PTA([model1(psr)])

	Nres1 = nestle_sample(pta1)
	
	print('For model 1 with modes 50,40,',k,', the log evidence is ',Nres1.logz, '+/-',Nres1.logzerr)
'''
