# CurrUncertEllipses
A library to fit uncertatinty ellipses around normally distributed current data, to allow characterisation and further analysis

```
Jonathan Tinker, Met Office Hadley Centre, 22/04/2022.
Based on the Appendix of Tinker et al. (2022).
Some analytical calculations have been undertaken with Wolfram Alpha (https://www.wolframalpha.com/).
```

## Introduction
We introduce a python toolbox for the analysis of bivariate normally distributed residual currents. This is largely based on fitting uncertainty ellipses around the mean, and using the size and shape of these ellipses to help describe the underlying data. Note that tidal currents are not normally distributed (as they are sinusoidal), so typically, the tide must be removed before this analysis is undertaken.


By fitting this ellipse to the data, we can use properties of the ellipse to help describe the underlying data. This methodology allows you to:
* assess whether a residual current is significantly different from zero, given the (e.g. inter-annual) variability.
* compare the residual currents of two model runs
* compare whether a single year is significantly different from a climatology.
* consider the likely range of current directions.



----

## Loading the toolbox
All the required functions are within `CurrUncertEllipses.py`.
`CurrUncertEllipses_examples.py` gives example analysis and figures.
`CurrUncertEllipses_ChiSqProb.py` provides tools to select the critical ellipse size (in standard deviations). Example data sets are given in `baroc_*.nc`.
Once downloaded, and copied to the correct location, the toolbox can be imported with:
```
import CurrUncertEllipses
```

## Data
The toolbox works on numpy arrays of the U and V component of the residual velocities. These should have three dimensions, with dimension `[0]` being time.
The github include example data sets (`baroc_*.nc`). `CurrUncertEllipses_examples.py` gives an example of how to load the example data set, with a loading function – the default data directory (datadir) is set as `./` – this may need to be changed for your system.

## Quick guide
```
#Data with time as Dimension[0], and latitude and longitude in dimensions [1] and [2].
# U_mat_1, V_mat_1, U_mat_2, V_mat_2
#Extract ellipse coefficients
n_std = 2.45
ellipse_dict = {}
ellipse_dict ['UV_1'] = ellipse_params_add_to_dict(ellipse_params(U_mat_1, V_mat_1, n_std=n_std))
ellipse_dict ['UV_2'] = ellipse_params_add_to_dict(ellipse_params(U_mat_2, V_mat_2, n_std=n_std))
#Compare two datasets
overlap_dict = overlapping_ellipse_area_from_dict( ellipse_dict['UV_1'],ellipse_dict['UV_2'])
OVL_dict = ellipse_overlap_coefficient_pdf_from_dict( ellipse_dict['UV_1'],ellipse_dict['UV_2'])
```

---

For further details refer to the attached [PDF](CurrUncertEllipses_documentation.pdf)
