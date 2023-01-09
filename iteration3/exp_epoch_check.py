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


#####Importing parfile and timfile#####


parfile = psrname+".DM1DM2.par"
timfile = psrname+".NB.tim"
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
#arn = red_noise_block(components=50)
#dmn = dm_noise_block(components=50)
#scn = chromatic_noise_block(gp_kernel='diag',idx=4, components=50)
           
model1 = ef + tm

pta1 = PTA([model1(psr)])

pta1.param_names



