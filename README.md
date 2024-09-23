# merger_precursor
This repository is a light curve model of precursors preceding mergers of SN progenitor star with compact object companion. These can be a possible explanation for long-rising precursors of supernovae, with timescales of many months to years. For details please see our paper (https://ui.adsabs.harvard.edu/abs/2024arXiv240612472T).

The python scripts compute the (i) mass-loss rate and velocity of the disk wind/(wind+CBO), and (ii) the luminosity and emission temperature of the precursor, powered by the disk wind only (f_Omega << 1 in the paper) and with additional reprocessing by the wind+CBO outflow. These are stored in the output files "lc_diskwind_(...).txt" (for the unimpeded disk wind), and "lc_CBO_(...).txt" (for the merged wind+CBO outflow). The calculations for each parameter should take typically a few to 10 minutes.

The prec_merger.py script is the script calculating the precursors for our helium star models, to reproduce the results in our paper. The prec_merger_flexible.py adds flexibility to the mass transfer history, with the mass transfer history assumed to be a power-law followed by a sharp rise to singularity (eq 43 in our paper). In this model the merger time, initial mass transfer rate, and the initial orbital period are free parameters, and can be specified by editing the script.

You may see NaN's in the color temperature/radius at the earliest times. This is when the CBO has not accumulated enough mass for a thermalization layer to develop, and we cannot reliably obtain the emission temperature (likely in the far UV). 

The current code deals with low-mass helium star models with 1.4 Msun NS companions, which are putative progenitors of Type Ibn SNe. To extend to other cases (IIn progenitors, BH companions etc.), new opacity tables and model grids of L2 mass loss (see Lu+23, MNRAS 519, 1409, and their source code) would be needed. We plan to add these capabilities in future projects.
