from enterprise.pulsar import Pulsar
from enterprise.signals.signal_base import PTA
from enterprise.signals.gp_signals import TimingModel
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise import constants as const
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_bases as gpb
from enterprise.signals import gp_priors as gpp
from enterprise.signals import (gp_signals, parameter, selections, utils,
                                white_signals)
from enterprise_extensions.blocks import white_noise_block, red_noise_block, dm_noise_block
from enterprise_extensions.chromatic.chromatic import dm_exponential_dip
from enterprise_extensions.blocks import chromatic_noise_block
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg as sl
import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import dynesty
from dynesty import plotting as dyplot
from dynesty.utils import resample_equal
from enterprise.signals.selections import custom_backends_dict, by_freq_band

#PLEASE GIVE PULSAR NAME HERE

psrname = "J1909-3744"

#creating a txt file for saving output
f= open("modes_arn_"+psrname+".txt","w+")

#####Importing parfile and timfile#####


parfile = "../"+psrname+".DM1DM2.par"
timfile = "../"+psrname+".final.tim"
psr = Pulsar(parfile, timfile,ephem='DE440')

tmin = psr.toas.min()
tmax = psr.toas.max()
Tspan = np.max(tmax) - np.min(tmin)
Tspan_years = Tspan/ 365.25 / 24 / 60 / 60
print(Tspan_years)  # time span of data in years



def by_groups(flags):
    """Selection function to split by -group flag"""
    flagvals = np.unique(flags["group"])
    return {val: flags["group"] == val for val in flagvals}

selection_by_groups = Selection(by_groups)

#White noise
efac = parameter.Uniform(0.1, 8)
equad = parameter.Uniform(-8,-2)

tm = TimingModel(use_svd=True, normed=True, coefficients=False)
ef = white_signals.MeasurementNoise(efac=efac, selection=selection_by_groups)
wn = white_signals.MeasurementNoise(efac=efac,log10_t2equad = equad, selection=selection_by_groups) 
          

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
    sampler = dynesty.NestedSampler(pta.get_lnlikelihood, prior_transform, len(pta.params),bound='multi')
    sampler.run_nested(dlogz = 0.1, print_progress=False )
    res = sampler.results
    return res


modes = [2,5,8,12]
rn = np.floor([i *Tspan_years for i in modes])   
rn = rn.astype(int)
print(rn)

for i in rn:
	arn = red_noise_block(Tspan=Tspan, components=i)
  
        # timing model
	tm = TimingModel(use_svd=True, normed=True, coefficients=False)
	
	model = wn + arn + tm

	pta = PTA([model(psr)])

	Dres = dynesty_sample(pta)
	
	print(" For arn model with arn modes as",i,", the log evidence is = {} ± {}".format(Dres.logz[-1], Dres.logzerr[-1]))
	
	print(" For arn model with arn modes as",i,", the log evidence is = {} ± {}".format(Dres.logz[-1], Dres.logzerr[-1]),file=f)
	

f.close()
