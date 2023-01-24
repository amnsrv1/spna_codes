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
timfile = "../"+psrname+".final.tim"
psr = Pulsar(parfile, timfile,ephem='DE440')

tmin = psr.toas.min()
tmax = psr.toas.max()
Tspan = np.max(tmax) - np.min(tmin)
Tspan_years = Tspan/ 365.25 / 24 / 60 / 60
print(Tspan_years)  # time span of data in years

#No. of modes for rednoises

arn_modes = 
dm_modes = 
sc_modes =

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
arn = red_noise_block(components=arn_modes)
dmn = dm_noise_block(components=dm_modes)
scn = chromatic_noise_block(gp_kernel='diag',idx=4, components=sc_modes)


#Please enter final model obtained from previous steps
model = wn/ef + arn + dmn + scn + tm  #As obtained in previous steps 

# initialize PTA
pta = PTA([model(psr)])


#USING PTMCMC (parameter estimation)

x0 = np.hstack([p.sample() for p in pta.params])
ndim = len(x0)
# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution

# where chains will be written to
outdir = 'output_param_est/'.format(str(psr.name))

chain = np.loadtxt(outdir + 'chain_1.txt')
# experiment with burn-in
pct = 0.3 # percent of the chain to toss
burn = int(pct * chain.shape[0])


samples = chain[burn:,:-4]

fig = corner.corner(samples,labels=list(pta.param_names), label_kwargs={"fontsize": 7},
                     quantiles=(0.16, 0.5, 0.84),show_titles=True)
plt.suptitle(psrname, fontsize=16)

plt.savefig(psrname+"_final.pdf")


#Plotting rednoise
ind_redA = list(pta.param_names).index(psrname+'_red_noise_log10_A')
ind_redgam = list(pta.param_names).index(psrname+'_red_noise_gamma')
corner.corner(chain[burn:, [ind_redgam, ind_redA]], 
                    labels=[r"$\gamma$", r"$\log_{10}(A_{red})$"],
                            label_kwargs={"fontsize": 12},
                            levels=[0.68,0.95], color='teal', show_titles=True);
plt.suptitle(psrname,x=0.75, fontsize=16)
plt.savefig(psrname+"_arn.pdf")


#Plotting dmnoise
ind_dmA = list(pta.param_names).index(psrname+'_dm_gp_log10_A')
ind_dmgam = list(pta.param_names).index(psrname+'_dm_gp_gamma')
corner.corner(chain[burn:, [ind_dmgam, ind_dmA]], 
                    labels=[r"$\gamma$", r"$\log_{10}(A_{dm})$"],
                            label_kwargs={"fontsize": 12},
                            levels=[0.68,0.95], color='teal', show_titles=True);
plt.suptitle(psrname,x=0.75, fontsize=16)
plt.savefig(psrname+"_dm.pdf")




#Plotting scnoise
ind_scA = list(pta.param_names).index(psrname+'_scat_gp_log10_A')
ind_scgam = list(pta.param_names).index(psrname+'_scat_gp_gamma')
corner.corner(chain[burn:, [ind_scgam, ind_scA]],
                    labels=[r"$\gamma$", r"$\log_{10}(A_{scat})$"],
                            label_kwargs={"fontsize": 12},
                            levels=[0.68,0.95], color='teal', show_titles=True);
plt.suptitle(psrname,x=0.75, fontsize=16)
plt.savefig(psrname+"_sc.pdf")

