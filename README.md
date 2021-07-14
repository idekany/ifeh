# Metallicity estimation of RR Lyrae stars from their I-band light curves

This repository contains the data files and codes for the basic workflow of the following study:
[Dekany, Grebel, & Pojmanski (2021)](http://arxiv.org/abs/2107.05983)

The codes have been developed under the following python environment:
 - `python` 3.8
 - `numpy` 1.19.5
 - `theano` 1.1.2
 - `PyMC3` 3.11.2
 - `arviz` 0.11.2
 - `matplotlib` 3.4.2
 - `scikit-learn` 0.24.2
 - `joblib` 0.24.2
 
 # Installation
 
 Simply copy the files into a desired folder and run the .py files with a suitable python interpreter (see above):
 
 `python <filename>.py`
 
 # Description of the files
 
 ### Source codes
 
 #### ifeh_io.py
 Contains static method (function) definitions required by the rest of the codes.
 
 #### ifeh_rfe_sfs.py
 Executable script for the computation of the optimal feature sets. See Sect. 3.1 of the paper for details.
 The input/output file names and parameters must be set in the source code's preamble.
 
 #### ifeh_mcmc_rrab.py
 Executable script for fitting the Bayesian regression model of the metallicity for the RRab stars. See Sect. 3.2 of the paper for details.
 The input/output file names and parameters must be set in the source code's preamble.
 
 #### ifeh_mcmc_rrc.py
 As above, but for the RRc stars.
 
 #### compute_mdfs.py
 Executable script for the computation of the RR Lyrae metallicity distribution functions (MDFs) of the following cosmic environments: inner and outer bulge,  bulge/disk transition region, southern Galactic disk, Sagittarius dwarf spheriodal galaxy, halo, Small and Large Magellanic Clouds. See Sect. 4 of the paper for details.
 The input/output file names and parameters must be set in the source code's preamble.
 
 ### Data files
 
 #### CHR_X_phot_single.dat, LIT_hires_feh_1_X_phot.dat, LIT_hires_feh_2_X_phot.dat
 
 Files containing individual spectroscopic metallicities and light-curve parameters of the objects in the development data set collected from the literature.
 See the column headers for more information. The columns `feh0`, `feh1`, `e_feh0`, `e_feh1` contain the metallicity values derived from the FeI and FeII lines, and their quoted uncertainties, respectively (where available). The columns `feh` and `e_feh` contain the metallicities and their unceratinties used in our study. Note that wherever possible, `e_feh` represents the square root of the pooled variance of the measurements computed for author+instrument pools (see Sect. 2.2 of the paper for details). Note that the column `shift` (see Sect. 2.2) must be added to `feh`. Gaia identifiers and coordinates correspond to Data Release 2 (DR2).
 
 #### o4rrab\*param.dat, o4rrc\*param.dat
 
 Files containing the I-band light-curve parameters of the RRab and RRc stars in the [OGLE Collection of Variable Stars](http://ogledb.astrouw.edu.pl/~ogle/OCVS/) including data from the OGLE-IV survey. See Sect. 2.1 of the paper for details. These data are used for the computation of the MDFs in Sect. 4.
 
 #### o4rrab_bulge_v_gpr_dff_param.dat
 
 V-band light-curve parameters of the bulge RRab stars in the [OGLE Collection of Variable Stars](http://ogledb.astrouw.edu.pl/~ogle/OCVS/) including data from the OGLE-IV survey. These data are used by `compute_mdfs.py` to make comparisons of the MDFs across various I- and V-band estimators.
 
