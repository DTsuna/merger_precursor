#
# A flexible merger-precursor light curve model, with tunable mass transfer history and mass ratio (q>0.4).
# For light curve model with the MESA models used and published in Tsuna+24, see prec_merger.py.
#

import math
import numpy as np
import sys

import constants as con
import functions as fun

# fixed model parameters (solar metallicity)
X_H = 0.0 # hydrogen mass fraction. currently only supports X_H=0.
CO = 'NS' # NS or BH
M_CO = 1.4*con.Msun # compact object mass

#
# adjustable model parameters
#

# for NS accretion disk
p_BB = 0.5 # power-law index of ADAF accretion disk (Blandford & Begelman 99)
rin_d = 1.2e6 # 6.*con.G*M_CO/con.c**2  # inner radius of ADAF accretion disk

#
# for mass transfer evolution
# The functional form assumed is assumed to be a power-law phase followed by a rise to a singularity:
# Mdot = -(Mdot_init) * (t_0/(t_0-t))^delta,
# with t_0 = P_init * Norb_mrg
# P_init sets the binary separation
# abin = [G*M_NS(1+1/q)*(P_init/2pi)**2]**(1/3)
#
q = 0.6 # fL2 models available from 0.4 to 1 in steps of 0.1 (1.4-3.5 Msun He star for a 1.4Msun NS.)
P_init = 100*86400. # initial orbital period (seconds)
Norb_mrg = 20 # how many orbits before merger (typically 10-100)
Mdot_init = 1e-3 # initial mass transfer rate in Msun/yr
delta = 1.5

# for CBO geometry
f_Omega_CBO = 1.0 # covering fraction of circumbinary outflow (0 means disk wind only, 1 means isotropic CBO)

##########################################################

#
# load and construct opacity tables
#

# load total opacity table (OPAL+low-T) and make interpolator
from load_kap_table import *
intp_lgkapscat = fun.load_kap_scat(X_H, 'inputs/opacity_tables/kappa_sc_z0.02_x0.0.txt') 

file_id = 'q%.1f_Pi%gday_Norb%g_Mdoti%g' % (q, P_init/86400, Norb_mrg, Mdot_init)

# mass transfer sampling is set to 200 points for speed
Nsamp_rlof = 200
t_merg = Norb_mrg*P_init
time = np.linspace(0., t_merg, Nsamp_rlof, endpoint=False) 
abin = np.ones(Nsamp_rlof) * (con.G*M_CO*(1.+1./q)*(P_init/2./math.pi)**2)**(1./3)
Mdot_rlof = Mdot_init*con.Msun/3.156e7 * (t_merg/(t_merg-time))**delta

rlof_dict = {'time': time, 'abin': abin, 'Mdot': Mdot_rlof}

# L2 mass-loss grid (Lu+23)
L2file = 'inputs/disk_sltns/M%.1f_interp/fL2grid_M1.4_q%.1f_case2.pkl' % (M_CO/con.Msun, q)

# calculate CO wind properties
print('working on %s, q=%g' % (file_id, q))
print('!!! Emission from disk wind !!!')
rout_d, Mdot_wind, Lwind, kappa_d = fun.COwind(L2file, q, time, abin, Mdot_rlof, p_BB, CO, M_CO, rin_d) 

# calculate disk wind emission temperature
Lobs, T_col, r_col, rtrap, vwind = fun.T_col_wind(time, Mdot_wind, rout_d, Lwind, intp_lgkapgrid, intp_lgkapscat)

# save disk wind parameters + observables
# do not save for first orbital period, as the RLOF
# has not become axisymmetric and calculations are less reliable
i_P = np.argmin(abs(time - P_init))

np.savetxt('lc_diskwind_%s.txt' % file_id, np.c_[(time-time[-1])[i_P:]/3.156e7, Mdot_wind[i_P:]*3.156e7/con.Msun, abin[i_P:], vwind[i_P:], Lobs[i_P:], T_col[i_P:], rtrap[i_P:], r_col[i_P:], rout_d[i_P:]], header='time [yr], Mdotwind [Msun/yr], abin [cm], vwind [cm/s], lum [erg/s], T_col [K], r_tr [cm], r_col [cm], r_sph [cm]', fmt='%.6g')

# include emission from the circum-binary outflow
if f_Omega_CBO > 0.0:
	print('!!! Emission from CBO !!!')
	# CBO velocity (assume 0.3 x orbital velocity)
	v_CBO = 0.3 * np.sqrt(con.G*(1.+1./q)*M_CO/abin)
	# get CBO mass-loss rate and velocity (in the absense of wind)
	Mdot_CBO = Mdot_rlof - Mdot_wind
	Mdot_total = Mdot_CBO + f_Omega_CBO * Mdot_wind
	vout = (Mdot_CBO * v_CBO + f_Omega_CBO * Mdot_wind * vwind) / Mdot_total
	
	Lobs_CBO, T_col, r_col, rtrap, vout_obs = fun.T_col_with_CBO(time, f_Omega_CBO, abin, Mdot_rlof, Mdot_wind, Mdot_total, v_CBO, vwind, Lobs, intp_lgkapgrid, intp_lgkapscat)

# save CBO parameters + observables
np.savetxt('lc_CBO_%s_fOmega%g.txt' % (file_id, f_Omega_CBO), np.c_[(time-time[-1])[i_P:]/3.156e7, Mdot_total[i_P:]*3.156e7/con.Msun, abin[i_P:], vout_obs[i_P:], Lobs_CBO[i_P:], T_col[i_P:], rtrap[i_P:], r_col[i_P:]], header='time [yr], MdotCBO [Msun/yr], abin [cm], v_CBO [cm/s], lum [erg/s], T_col_CBO [K], r_tr [cm], r_col [cm]', fmt='%.6g')
