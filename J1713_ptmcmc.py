
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
sc_modes = 30
crn_modes = 100

##### Set up signals #####


parfile = "J1713+0747.par"
timfile = "J1713+0747.NB.tim"



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


# In[18]:



arn = red_noise_block(psd='powerlaw', prior='log-uniform', Tspan=Tspan,
                    components=arn_modes, gamma_val=None, coefficients=False,
                    select=None, modes=None, wgts=None, combine=True,
                    break_flat=False, break_flat_fq=None,
                    logmin=None, logmax=None, dropout=False, k_threshold=0.5)


# In[23]:


scn = chromatic_noise_block(gp_kernel='diag', psd='powerlaw',
                          nondiag_kernel='periodic',
                          prior='log-uniform', dt=15, df=200,
                          idx=4, include_quadratic=False,
                          Tspan=Tspan, name='scat', components=sc_modes,
                          coefficients=False)


# In[24]:


dmn = dm_noise_block(gp_kernel='diag', psd='powerlaw', nondiag_kernel='periodic',
                   prior='log-uniform', dt=15, df=200,
                   Tspan=Tspan, components=dm_modes,
                   gamma_val=None, coefficients=False)


# In[25]:


crn = chromatic_noise_block(gp_kernel='diag', psd='powerlaw',
                          nondiag_kernel='periodic',
                          prior='log-uniform', dt=15, df=200,
                          idx=chr_idx,Tspan=Tspan, name='chrom', components=crn_modes,
                          coefficients=False)


# In[26]:



# timing model
tm = gp_signals.TimingModel(use_svd=True)

#LET US DEFINE 5 DIFFERENT MODEL COMBINATIONS TO COMPARE THEIR EVIDENCES

'''
model1 = ef + tm
model2 = wn + tm
model3 = ef + arn + tm
model4 = wn + arn + tm

model5 = wn + arn + crn + tm
'''
model6 = ef + dmn + scn + tm 

#full model is sum of components
#model = ef + eq + crn +dmn + scn + tm  


# In[188]:


# initialize PTA
#pta = signal_base.PTA([model(psr)])
#pta1 = signal_base.PTA([model1(psr)])
#pta2 = signal_base.PTA([model2(psr)])
#pta3 = signal_base.PTA([model3(psr)])
#pta4 = signal_base.PTA([model4(psr)])
#pta5 = signal_base.PTA([model5(psr)])
pta6 = signal_base.PTA([model6(psr)])



#USING PTMCMC (parameter estimation)

x0 = np.hstack([p.sample() for p in pta6.params])
ndim = len(x0)
# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2) # helps to tune MCMC proposal distribution

# where chains will be written to
outdir = 'output/'.format(str(psr.name))


# sampler object
sampler = ptmcmc(ndim, pta6.get_lnlikelihood, pta6.get_lnprior, cov,
                 outDir=outdir, 
                 resume=False)


# In[27]:


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



chain = np.loadtxt(outdir + 'chain_1.txt')
# experiment with burn-in
pct = 0.3 # percent of the chain to toss
burn = int(pct * chain.shape[0])


psrj = psr.name



samples = chain[burn:,:-4]

fig = corner.corner(samples,
                     quantiles=(0.16, 0.5, 0.84),labels = ['1460_100_b1_pre36_efac',
 '1460_200_post36_efac',
 '1460_200_pre36_efac',
 '500_100_pre36_efac',
 '500_200_post36_efac',
 '500_200_pre36_efac',
 'scat_gamma','scat_log10_A',
 'dm_gamma',
 'dm_log10_A'])
plt.suptitle("J1713+0747", fontsize=16)

plt.savefig("J1713+0747_dmscn.pdf")

'''
#Plotting rednoise
ind_redA = list(pta6.param_names).index(psrj+'_red_noise_log10_A')
ind_redgam = list(pta6.param_names).index(psrj+'_red_noise_gamma')
corner.corner(chain[burn:, [ind_redgam, ind_redA]], 
                    labels=[r"$\gamma$", r"$\log_{10}(A_{red})$"],
                            label_kwargs={"fontsize": 12},
                            levels=[0.68,0.95], color='teal', show_titles=True);
plt.suptitle("J1909-3744",x=0.75, fontsize=16)
plt.savefig("J1909inpta5040_arn.pdf".format(psrj))
'''

#Plotting dmnoise
ind_dmA = list(pta6.param_names).index(psrj+'_dm_gp_log10_A')
ind_dmgam = list(pta6.param_names).index(psrj+'_dm_gp_gamma')
corner.corner(chain[burn:, [ind_dmgam, ind_dmA]], 
                    labels=[r"$\gamma$", r"$\log_{10}(A_{dm})$"],
                            label_kwargs={"fontsize": 12},
                            levels=[0.68,0.95], color='teal', show_titles=True);
plt.suptitle("J1713+0747",x=0.75, fontsize=16)
plt.savefig("J1713inpta4030_dm.pdf".format(psrj))





#Plotting scattering noise
ind_scA = list(pta6.param_names).index(psrj+'_scat_gp_log10_A')
ind_scgam = list(pta6.param_names).index(psrj+'_scat_gp_gamma')
corner.corner(chain[burn:, [ind_scgam, ind_scA]],
                    labels=[r"$\gamma$", r"$\log_{10}(A_{scat})$"],
                            label_kwargs={"fontsize": 12},
                            levels=[0.68,0.95], color='teal', show_titles=True);
plt.suptitle("J1713+0747",x=0.75, fontsize=16)
plt.savefig("J1713inpta4030_sc.pdf".format(psrj))


