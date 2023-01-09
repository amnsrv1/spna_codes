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


parfile = "../"+psrname+".DM1DM2.par"
timfile = "../"+psrname+".NB.tim"
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
arn = red_noise_block(components=50)
dmn = dm_noise_block(components=50)
scn = chromatic_noise_block(gp_kernel='diag',idx=4, components=50)
           
model1 = ef + tm
model2 = wn + tm

# initialize PTA

pta1 = PTA([model1(psr)])
pta2 = PTA([model2(psr)])



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
    sampler=dynesty.NestedSampler(pta.get_lnlikelihood,prior_transform,len(pta.params),bound='multi')
    sampler.run_nested(dlogz = 0.1, print_progress=False )
    res = sampler.results
    return res



#creating a txt file for saving output
f= open("equad_check_"+psrname+".txt","w+")



#Dynesty sampler run and plotting
Dres1 = dynesty_sample(pta1)
print("For model 1, log evidence = {} ± {}".format(Dres1.logz[-1], Dres1.logzerr[-1]))
print("For model 1, log evidence = {} ± {}".format(Dres1.logz[-1], Dres1.logzerr[-1]), file = f)
weights1 = np.exp(Dres1['logwt'] - Dres1['logz'][-1])
samples1 = resample_equal(Dres1.samples, weights1)
fig1 = corner.corner(samples1,labels=list(pta1.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model1.pdf')




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

