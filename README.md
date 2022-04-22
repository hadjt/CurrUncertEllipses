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

For further details refer to the attached [PDF](CurrUncertEllipses_documentation.pdf)
