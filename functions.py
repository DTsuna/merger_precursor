import math
import numpy as np
from scipy.interpolate import RectBivariateSpline
import pickle

import sys
import constants as con

#
# opacity table manipulations/referencing
#

# load scattering opacity table and make interpolator
def load_kap_scat(X_H, kap_scat_file):
	assert X_H==0, "only supports no-H gas."
	data = np.loadtxt(kap_scat_file)
	lgrho = data[1:,0]
	lgT = data[0,1:]
	lgkap = data[1:,1:]
	intp_lgkapscat = RectBivariateSpline(lgrho, lgT, lgkap, kx=3, ky=3, s=0)
	return intp_lgkapscat

# get total opacity from table
def kap_tot_intp(rho, T, intp_lgkap):
	lgT = math.log10(T)
	lgrho = math.log10(rho)
	lgR = lgrho - 3*lgT + 18
	return 10**intp_lgkap(lgR, lgT)

# get scattering opacity from table
def kap_s_intp(rho, T, intp_lgkapscat):
	return 10**intp_lgkapscat(math.log10(rho), math.log10(T))

# approx. scattering opacity, assuming fully ionized gas # NOTE not used in this model
def kap_s_approx(X_H):
	return 0.2*(1.+X_H)

# approx. total opacity (above + Kramer's law) # NOTE not used in this model
def kap_tot_approx(X_H, rho, T):
	return kap_s_approx(X_H) + 8e23*(1.+X_H)*rho*T**(-3.5)


#
# inner accretion disk(+wind) modeling
#

def get_fL2grid(pkl_file):
	# L2 mass loss from Model of Lu, Fuller, Quataert, Bonnerot 23, MNRAS 519, 1409
	# Updates from their (public) source code:
	# - expanded grid: 1.4Rsun < a_bin < 1800Rsun, 5e-6 Msun/yr < Mdot_RLOF < 5e-1 Msun/yr
	# - added kappa in the output (for calculating disk wind, see function rout_d)
	dbfile = open(pkl_file, 'rb')
	db = pickle.load(dbfile)
	Mdot_arr = 10**db[1] # db[1]: log10_Msun/yr
	abin_arr = 10**db[2] # db[2]: log10_Rsun
	fL2out_arr = 10**db[3] # db[3]: dimensionless
	log10kappa_arr = db[-1]
	return Mdot_arr, abin_arr, fL2out_arr, log10kappa_arr


# outermost outflowing disk radii
def rout_d(Mdotacc, q, abin, kappa):
	lgq = math.log10(q)
	xL1 = -0.0355 * lgq**2 + 0.251 * abs(lgq) + 0.500
	r_circ = (1. - xL1)**4 * abin / (q/(1.+q))
	r_sph = Mdotacc * kappa / (4.*math.pi*con.c)
	return np.minimum(r_sph, r_circ)

# calculate CO wind properties
def COwind(L2file, q, time, abin, Mdot, p_BB, CO, MCO, r_in):
	# load Lu+23 L2 mass-loss results
	Mdot_arr, abin_arr, fL2out_arr, log10kappa_arr = get_fL2grid(L2file)
	# approximate a_bin and q as constant throughout RLOF, with initial values
	a_bin_init = abin[0]/con.Rsun
	i_a_bin = np.argmin(abs(abin_arr - a_bin_init))
	# obtain Mdotacc (accretion onto NS) as a function of Mdot (mass transfer rate), for a given a_bin
	log10Mdot_rlof = np.log10(Mdot*3.156e7/con.Msun)  # g/s -> Msun/yr
	# Mdot_arr, Mdotacc_arr are 1D arrays in units of Msun/yr.
	log10Mdot_arr = np.log10(Mdot_arr)
	log10Mdotacc_arr = log10Mdot_arr + np.log10(1.-fL2out_arr[:,i_a_bin])
	# NOTE: interpolate, but using the edge value once outside the defined range of Mdot_rlof
	Mdotacc = 10 ** np.array([np.interp(x, log10Mdot_arr, log10Mdotacc_arr) for x in log10Mdot_rlof])
	kappa_disk = 10 ** np.array([np.interp(x, log10Mdot_arr, log10kappa_arr[:,i_a_bin]) for x in log10Mdot_rlof])
	Mdotacc *= con.Msun/3.156e7 # Msun/yr -> g/s
	# outer disk wind radii
	r_out = rout_d(Mdotacc, q, a_bin_init * con.Rsun, kappa_disk)
	# use outer and inner radii to get wind mass-loss rate and luminosity (e.g. Tsuna+24, arXiv:2401.02389)
	Mdot_wind = Mdotacc * (1.0 - (r_in/r_out)**p_BB)
	L_wind = 0.5*Mdotacc* p_BB/(1.-p_BB) * con.G * MCO/r_in * ((r_in/r_out)**p_BB - (r_in/r_out))
	if CO == 'NS':
		# dissipation by accretion onto the NS
		L_acc = 0.5*(Mdotacc*(r_in/r_out)**p_BB)*con.G*MCO/r_in
	elif CO == 'BH':
		L_acc = 0.0
	else:
		raise ValueError("invalid compact object %s, has to be NS or BH" % CO)
	return r_out, Mdot_wind, L_wind + L_acc, kappa_disk


def wind_lum_vel(kappa, r_out, Mdot_wind, L_wind): 
	# trapping radii of the disk wind
	r_trap = r_out * (1. + kappa * Mdot_wind/(4.*math.pi*r_out*con.c))
	# some fraction goes to radiation and rest goes to wind kinetic energy
	L_obs = L_wind * (r_out/r_trap)**(2./3.)
	v_wind = np.sqrt(2.*L_wind/Mdot_wind * (1.-(r_out/r_trap)**(2./3.)))
	return r_trap, L_obs, v_wind


def CBO_lum_vel(kappa, f_Omega, abin, MdotRLOF, Mdot_wind, v_CBO, vwind, Lobs_wind):
	# get CBO mass-loss rate and velocity (in the absense of wind)
	Mdot_CBO = MdotRLOF - Mdot_wind
	Mdot_tot = Mdot_CBO + f_Omega * Mdot_wind
	# momentum conservation
	vout = (Mdot_CBO * v_CBO + f_Omega * Mdot_wind * vwind) / Mdot_tot
	Lsh = 0.5*f_Omega*Mdot_wind*vwind**2 + 0.5*Mdot_CBO*v_CBO**2 - 0.5*(Mdot_CBO+f_Omega*Mdot_wind)*vout**2
	assert Lsh > 0.0
	L_in = Lobs_wind + Lsh
	# trapping radii of the disk wind
	r_trap = abin * (1. + kappa * Mdot_tot/(4.*math.pi*abin*con.c))
	# some fraction goes to radiation and rest goes to wind kinetic energy
	L_obs = (Lobs_wind + Lsh) * (abin/r_trap)**(2./3.)
	vout_obs = np.sqrt(vout**2 + 2.*(Lobs_wind+Lsh)/Mdot_tot * (1.-(abin/r_trap)**(2./3.)))
	return r_trap, L_obs, vout_obs


#
# wind density and temperature profiles
#


# absorption and total opacity for table-interpolation. Appendix of Tsuna+24, arXiv:2406.12472
def kappa_abs_tot(rho, T, intp_lgkap, intp_lgkapscat):
	logrho = math.log10(rho)
	logT = math.log10(T)
	lgR = logrho - 3*logT + 18
	kap_s = kap_s_intp(rho, T, intp_lgkapscat) 
	if lgR < -8:
		# if lgR < -8 (outside table), obtain kap_abs by extrapolation (independent of rho)
		den_edge = 10**(-26+3*logT) 
		kap_abs_edge = kap_tot_intp(den_edge, T, intp_lgkap) - kap_s_intp(den_edge, T, intp_lgkapscat)
		kap_abs = max(0.,kap_abs_edge) # for kap\propto rho, use: 10**(lgR+8) * max(0.,kap_abs_edge)
		kap_tot = kap_s + kap_abs
	elif logT < 3.3: # neglect dust opacity
		kap_abs = 0.0
		kap_tot = kap_s
	else:
		kap_tot = kap_tot_intp(rho, T, intp_lgkap)
		kap_abs = max(0., kap_tot - kap_s)
	return kap_abs, kap_tot


# density/velocity profile at time t
def rho_v_prof(r_array, t, tzero_arr, rwind_array, Mdot_of_tzero, vwind_of_tzero):
	# rho(r,t) = Mdot(t0)/(4pi r^2 v_w(t0))
	# t0(r,t) is solved by r = r_0 + v_w(t0) (t-t_0)
	rho_array = np.zeros(len(r_array))
	v_array = np.zeros(len(r_array))
	for i, r in enumerate(r_array):
		# obtain tzero at this r for given time.
		this_tzero = np.interp(r, np.flip(rwind_array), np.flip(tzero_arr)) 
		# interpolation of log10 as mass-loss rate changes exponentially at ramp-up phase.
		Mdot_this_tzero = 10** np.interp(this_tzero, tzero_arr, np.log10(Mdot_of_tzero))
		vwind_this_tzero = np.interp(this_tzero, tzero_arr, vwind_of_tzero)
		rho_array[i] = Mdot_this_tzero / (4.*math.pi*r**2*vwind_this_tzero)
		v_array[i] = vwind_this_tzero
	return rho_array, v_array


# temperature profile solving flux-limited diffusion approximation 
def T_prof(r_array, rho_array, Lobs, intp_lgkap, intp_lgkapscat):
	# construct temperature profile from density profile, Lobs
	# NOTE r_array is decreasing array (r_array[0] is biggest)
	T_array = np.zeros(len(r_array))
	# edge is simply from L=4pi*r^2*acT^4
	T_array[0] = (Lobs/4./math.pi/r_array[0]**2/con.a_rad/con.c)**0.25
	# initially assume T^4 \propto r^(-2) -> dT^4/dr = -2*T^4/r 
	dT4dr = -2.*T_array[0]**4/r_array[0]
	for i in range(1, len(r_array)):
		r = r_array[i]
		dr = r_array[i-1]-r_array[i] # r[i-1]>r[i], so dr>0
		#
		# flux-limited diffusion
		# lmbd = (2+R)/(6+3R+R^2)
		# lmbd * R = L/(4pi r^2 acT^4)
		# - R->0, lmbd->1/3 for optically thick (diffusion approximation)
		# - R->inf, lmbd->1/R for optically thin (L=4pi r^2 acT^4)
		#
		# Assume values for T_array[0] and d(T^4)/dr[0] (optically thin limit)
		# -> get R and lambda, solve flux limited diffusion and get d(T^4)/dr (<0) inwards
		# -> (T_array[1])^4 = (T_array[0]) + d(T^4)/dr * (-dr) (minus because you're moving inwards; careful with sign)
		# -> get R with this d(T^4), get lambda, solve flux limited diffusion and get d(T^4)/dr
		# -> ...
		#
		# get density and Rosseland mean
		rho = rho_array[i]
		dummy, kap_R = kappa_abs_tot(rho, T_array[i-1], intp_lgkap, intp_lgkapscat)
		R = (-dT4dr)/(kap_R*rho*T_array[i-1]**4)
		assert R > 0., "R=%g, dT4dr=%g, Lobs=%g" % (R, dT4dr, Lobs)
		lmbd = (2.+R)/(6.+3.*R+R*R)
		# flux-limited diffusion.
		dT4dr = -(kap_R*rho*Lobs)/lmbd/(4.*math.pi*r**2*con.a_rad*con.c)
		T4 = T_array[i-1]**4 - dr*dT4dr
		T_array[i] = T4**0.25
	return T_array, kap_R


#
# obtain color temperature of wind emission
#

def T_col_wind(time, Mdot_wind, rout, L_wind, intp_lgkap, intp_lgkapscat):
	T_col_arr = np.zeros(len(time))
	r_col_arr = np.zeros(len(time))
	Lobs = np.zeros(len(time))
	rtrap = np.zeros(len(time))
	vwind = np.zeros(len(time))
	for i,t in enumerate(time):
		# Obtain trapping radius, luminosity and temperature profile self-consistently.
		# We first make an initial guess of kappa at trapping radius
		# -> obtain r_trap, Lobs, v_wind
		# -> obtain temperature profile using flux-limtied diffusion
		# -> check if the kappa at r_trap is converged. if not, try again with this newly obtained kappa.
		# -> ....
		#
		# initial guess of kappa
		kappa = 0.2
		kappa_err = 1.0
		while (kappa_err > 1e-2):
			last_kappa = kappa
			rtrap[i], Lobs[i], vwind[i] = wind_lum_vel(kappa, rout[i], Mdot_wind[i], L_wind[i])
			this_rtr = rtrap[i]
			rwind_array = np.array([(rtrap[k]+vwind[k]*(t-t0)) for k,t0 in enumerate(time)])
			# outermost radii of wind 
			r_wind = rwind_array[0]
			# NOTE r sampled outside-in for optical depth calculation
			r_array = np.logspace(math.log10(r_wind), math.log10(this_rtr), 5000)
			# construct density and temperature profile
			rho_array, v_array = rho_v_prof(r_array, t, time, rwind_array, Mdot_wind, vwind)
			T_array, kappa = T_prof(r_array, rho_array, Lobs[i], intp_lgkap, intp_lgkapscat) 
			kappa = 0.5*(kappa+last_kappa)
			kappa_err = abs((kappa - last_kappa)/kappa)
		# temperature profile solution converged within 1%
		print('time from merger=%eyr, r_tr=%2ecm, kap_tr=%e' % ((t-time[-1])/3.156e7, rtrap[i], kappa))	
		tau_eff = 0.0
		r_col = -1.0
		for j, r in enumerate(r_array[1:]):
			tau_eff_old = tau_eff
			thisdr = r_array[j] - r
			thisrho = rho_array[j+1]
			thisT = T_array[j+1]
			dT = T_array[j] - thisT
			this_kabs, this_ktot = kappa_abs_tot(thisrho, thisT, intp_lgkap, intp_lgkapscat)
			this_keff = math.sqrt(3.*this_kabs*this_ktot)
			tau_eff += this_keff * thisrho * thisdr
			if tau_eff_old < 1.0 and tau_eff > 1.0:
				frac = (tau_eff - 1.0) / (tau_eff - tau_eff_old)
				r_col = r + frac * thisdr
				r_col_arr[i] = r_col
				T_col_arr[i] = thisT + frac * dT
				break
		# if we can't find color radius (can't thermalize, set temp to nan)
		if r_col < 0:
			T_col_arr[i] = np.nan
			r_col_arr[i] = np.nan
	return Lobs, T_col_arr, r_col_arr, rtrap, vwind


def T_col_with_CBO(time, f_Omega, abin, Mdot_rlof, Mdot_wind, Mdot_tot, v_CBO, vwind, Lobs_wind, intp_lgkap, intp_lgkapscat):
	T_col_arr = np.zeros(len(time))
	r_col_arr = np.zeros(len(time))
	Lobs = np.zeros(len(time))
	rtrap = np.zeros(len(time))
	v_out = np.zeros(len(time))
	for i,t in enumerate(time):
		# Obtain trapping radius, luminosity and temperature profile self-consistently. See above T_col_wind
		# initial guess of kappa
		kappa = 0.2
		kappa_err = 1.0
		while (kappa_err > 1e-2):
			last_kappa = kappa
			this_abin = abin[i]
			rtrap[i], Lobs[i], v_out[i] = CBO_lum_vel(kappa, f_Omega, this_abin, Mdot_rlof[i], Mdot_wind[i], v_CBO[i], vwind[i], Lobs_wind[i])
			rCBO_array = np.array([(rtrap[k]+v_out[k]*(t-t0)) for k,t0 in enumerate(time)])
			# outermost radii of CBO 
			r_CBO = rCBO_array[0]
			# NOTE r sampled outside-in for optical depth calculation
			r_array = np.logspace(math.log10(r_CBO), math.log10(this_abin), 5000)
			# construct density and temperature profile
			rho_array, v_array = rho_v_prof(r_array, t, time, rCBO_array, Mdot_tot, v_out)
			T_array, kappa = T_prof(r_array, rho_array, Lobs[i], intp_lgkap, intp_lgkapscat)
			kappa = 0.5*(kappa+last_kappa)
			kappa_err = abs((kappa - last_kappa)/kappa)
		# temperature profile solution converged within 1%
		print('time from merger=%eyr, r_tr=%2ecm, kap_tr=%e' % ((t-time[-1])/3.156e7, rtrap[i], kappa))	
		tau_eff = 0.0
		r_col = -1.0
		for j, r in enumerate(r_array[1:]):
			tau_eff_old = tau_eff
			thisdr = r_array[j] - r
			thisrho = rho_array[j+1]
			thisT = T_array[j+1]
			dT = T_array[j] - thisT
			this_kabs, this_ktot = kappa_abs_tot(thisrho, thisT, intp_lgkap, intp_lgkapscat)
			this_keff = math.sqrt(3.*this_kabs*this_ktot)
			tau_eff += this_keff * thisrho * thisdr
			if tau_eff_old < 1.0 and tau_eff > 1.0:
				frac = (tau_eff - 1.0) / (tau_eff - tau_eff_old)
				r_col = r + frac * thisdr
				r_col_arr[i] = r_col
				T_col_arr[i] = thisT + frac * dT
				break
		# if we can't find color radius (can't thermalize, set temp to nan)
		if r_col < 0:
			T_col_arr[i] = np.nan
			r_col_arr[i] = np.nan
	return Lobs, T_col_arr, r_col_arr, rtrap, v_out
