# merger_precursor
This repository shows a light curve model of precursors preceding mergers of SN progenitor star with compact object companion.

The prec_merger.py script is the execution script, which computes the (i) mass-loss rate and velocity of the disk wind/(wind+CBO), and (ii) the luminosity and temperature of the precursor, powered by the disk wind only (f_Omega << 1 in the paper) and with additional reprocessing by the wind+CBO outflow. These are stored in the output files "lc_diskwind_(...).txt" (for the unimpeded disk wind), and "lc_CBO_(...).txt" (for the merged wind+CBO outflow). The calculations for each parameter should take typically a few to 10 minutes.

Currently the inputs are hard-coded to be usable for only the He-NS binaries in the paper, to reproduce the results in the paper. The code will be updated upon acceptance to also be able to handle more flexible mass-transfer histories and mass ratios.
