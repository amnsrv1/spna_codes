
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


#PLEASE GIVE PULSAR NAME HERE

psrname = "J1909-3744"

#creating a txt file for saving output
f= open("model_selection_"+psrname+".txt","w+")


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
Tspan_chrom_years = 4*Tspan_years   # 4 times Tspan for DM and Scat noises

#No. of modes for rednoises

arn_modes = np.floor(Tspan_years*12)  ##taking once per month as max cadence
dm_modes =  np.floor(Tspan_chrom_years*12)
sc_modes =  np.floor(Tspan_chrom_years*12)
print ('arn modes, dmmodes and scmodes used are' , arn_modes, dm_modes, sc_modes,file = f)


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
ecorr = parameter.Uniform(-8, -2)

#White noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection_by_groups)
wn = white_signals.MeasurementNoise(efac=efac,log10_t2equad = equad, selection=selection_by_groups) 
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection_by_groups)


arn = red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan,
                    components=arn_modes, gamma_val=None, coefficients=False,
                    select=None, modes=None, wgts=None, combine=True,
                    break_flat=False, break_flat_fq=None,
                    logmin=None, logmax=None, dropout=False, k_threshold=0.5)



scn = chromatic_noise_block(gp_kernel='diag', psd='powerlaw',
                          nondiag_kernel='periodic',
                          prior='log-uniform', dt=15, df=200,
                          idx=4, include_quadratic=False,
                          Tspan=Tspan_chrom, name='scat', components=sc_modes,
                          coefficients=False)


dmn = dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', dt=15, df=200,
                   Tspan=Tspan_chrom, components=dm_modes,
                   gamma_val=None, coefficients=False)



# timing model
tm = gp_signals.TimingModel(use_svd=True, normed=True, coefficients=False)

#LET US DEFINE 4 DIFFERENT MODEL COMBINATIONS TO COMPARE THEIR EVIDENCES

model1 = wn +ec + tm
model2 = wn + ec + arn + tm
model3 = wn + ec + arn + dmn +  tm
model4 = wn + ec + dmn + scn + tm
model5 = wn + ec + arn + dmn + scn + tm
model6 = wn + ec + dmn + tm


# initialize PTA

pta1 = signal_base.PTA([model1(psr)])
pta2 = signal_base.PTA([model2(psr)])
pta3 = signal_base.PTA([model3(psr)])
pta4 = signal_base.PTA([model4(psr)])
pta5 = signal_base.PTA([model5(psr)])
pta6 = signal_base.PTA([model6(psr)])

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




Dres5 = dynesty_sample(pta5)
print("For model 5, log evidence = {} ± {}".format(Dres5.logz[-1], Dres5.logzerr[-1]), file = f)
weights5 = np.exp(Dres5['logwt'] - Dres5['logz'][-1])
samples5 = resample_equal(Dres5.samples, weights5)
fig5 = corner.corner(samples5,labels=list(pta5.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model5.pdf')



Dres6 = dynesty_sample(pta6)
print("For model 6, log evidence = {} ± {}".format(Dres6.logz[-1], Dres6.logzerr[-1]), file = f)
weights6 = np.exp(Dres6['logwt'] - Dres6['logz'][-1])
samples6 = resample_equal(Dres6.samples, weights6)
fig6 = corner.corner(samples6,labels=list(pta6.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.savefig(psrname+'_model6.pdf')



et = time.time()
ep_time = (et-st)
print('elapsed time:', ep_time, 's')  


f.close()

