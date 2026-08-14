### XXN scripts for sharing

#### rx5day_method_plots.ipynb
First stage of selection with plots to show method

#### rx5day_clim_identify.py
Event identification script, reads in whole of CESM2 LENS for region specified. Run from command line with inputs for sigma level, region: e.g. 6sigma over norway: python rx5day_clim_identify.py 6 'NO_ext'

#### rx5day_clim_identify.ipynb
Same as above but in notebook for testing, reads in first 5 years and 5 ensemble members only. NB probably not up to date with above script

#### categorise_imgs.ipynb
clustering algorithm for event output plots, needs keras environment (see keras-env.yml)

#### plot_grouped_categorised_events.ipynb
more plotting for clusters for grouped events (also needs keras env)

#### keras-env.yml
Example keras env for categorise_imgs script.

### archive

#### 14day_method_testing.ipynb
Identify events in large ensemble, relative to climatology baseline. Uses 14day mean temperature. Adapt units etc and sum over 5day for precip and Rx5day.

#### plot_grouped_events.ipynb
Plotting clustered events (after running categorisation script)

#### rx5day_clim_identify.py
Full rx5day identifying events script.

#### precip_identify.py
OLD script for precip event identification, not up to date/some errors.
