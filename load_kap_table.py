# based on Lu+23 code, with small modifications
import numpy as np
from scipy.interpolate import RectBivariateSpline
from math import exp

fname1 = './inputs/opacity_tables/gn93_z0.02_x0.0_upd.data' # updated: removed lowest T's (unreliable for helium-dominated gas)
fname2 = './inputs/opacity_tables/lowT_fa05_gn93_z0.02_x0.0.data'


def parse(fname):
    lgRarr = np.loadtxt(fname, max_rows=1, skiprows=5, unpack=True, dtype=float)
    data = np.loadtxt(fname, skiprows=7, unpack=True, dtype=float)
    lgTarr = data[0]
    NR = len(lgRarr)
    NT = len(lgTarr)
    lgkap = np.zeros((NR, NT), dtype=float)
    for i in range(NR):
        for j in range(NT):
            lgkap[i, j] = data[i+1][j]
    return lgRarr, lgTarr, lgkap


def sigm(x, x0, dx):   # sigmoid function between 0 and 1
    arg = (x-x0)/dx
    if arg < -50:
        return 0.
    if arg > 50:
        return 1.
    else:
        return exp(arg)/(exp(arg) + 1)


# ----- highT opacity table 1
lgR1, lgT1, lgkap1 = parse(fname1)
lgRmin, lgRmax = min(lgR1), max(lgR1)
lgTmin1, lgTmax1 = min(lgT1), max(lgT1)
intp_lgkap1 = RectBivariateSpline(lgR1, lgT1, lgkap1, kx=3, ky=3, s=0)

# ----- lowT opacity table 2 (which has the same lgRmin and lgRmax)
lgR2, lgT2, lgkap2 = parse(fname2)
lgTmin2, lgTmax2 = min(lgT2), max(lgT2)
intp_lgkap2 = RectBivariateSpline(lgR2, lgT2, lgkap2, kx=3, ky=3, s=0)
lgT_blend_center = (lgTmax2 + lgTmin1)/2
dlgT_blend = (lgTmax2-lgTmin1)/10   # width for the sigmoid function


# full grid to interpolate upon
NlgR_old, NlgT_old = 200, 200
lgRgrid_old = np.linspace(lgRmin, lgRmax, NlgR_old, endpoint=True)
lgTgrid_old = np.linspace(min(lgTmin1, lgTmin2), max(lgTmax1, lgTmax2),
                          NlgT_old, endpoint=True)
lgkapgrid_old = np.zeros((NlgR_old, NlgT_old), dtype=float)

for i in range(NlgR_old):
    x = lgRgrid_old[i]
    for j in range(NlgT_old):
        y = lgTgrid_old[j]
        if y < lgTmin1:  # use linear extrapolation
            yleft = lgT1[0]
            yright = lgT1[1]
            zleft = intp_lgkap1(x, yleft)[0][0]
            zright = intp_lgkap1(x, yright)[0][0]
            slope = (zright - zleft)/(yright - yleft)
            z1 = zleft + slope * (y - yleft)
        else:
            z1 = intp_lgkap1(x, y)[0][0]
        if y > lgTmax2:  # use linear extrapolation
            yleft = lgT2[-1]
            yright = lgT2[-2]
            zleft = intp_lgkap2(x, yleft)[0][0]
            zright = intp_lgkap2(x, yright)[0][0]
            slope = (zright - zleft)/(yright - yleft)
            z2 = zleft + slope * (y - yleft)
        else:
            z2 = intp_lgkap2(x, y)[0][0]
        weight = sigm(y, lgT_blend_center, dlgT_blend+1e-10) # small number to silence warning if dlgT_blend=0
        # smoothly connect these two functions
        # such that z ~ z2 if y < lgT_blend_center
        #           z ~ z1 if y > lgT_blend_center
        z = weight * z1 + (1 - weight) * z2
        lgkapgrid_old[i, j] = z

lgRgrid = lgRgrid_old
lgTgrid = lgTgrid_old
lgkapgrid = lgkapgrid_old
intp_lgkapgrid = RectBivariateSpline(lgRgrid, lgTgrid, lgkapgrid, kx=3, ky=3, s=0)
