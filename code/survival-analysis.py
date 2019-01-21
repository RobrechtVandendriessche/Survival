# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 15:24:17 2019

@author: vande70
"""

import os
print(os.getcwd())
#os.chdir('OneDrive\Cursussen\Programmeren\Python\Kaggle_housing-prices'); print os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lifelines.datasets import load_waltons
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter


#sns.palplot(sns.color_palette(color_scheme))
sns.set_style('whitegrid')


### Importing and inspecting the data

# Import
df = load_waltons()

# First inspection
print(df.head())
print(df.describe())
print(df.group.value_counts())

T = df["T"]     # Durations 
E = df["E"]     # Events

# Plot population Kaplan-Meier function
kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)
kmf.survival_function_
kmf.median_
kmf.plot()

# Plot Kaplan-Meier function per group
groups = df['group']
ix = (groups == "miR-137")
ix.value_counts()
kmf.fit(T[~ix], E[~ix], label='control')
ax = kmf.plot()

kmf.fit(T[ix], E[ix], label='miR-137')
ax = kmf.plot(ax=ax)

# Plot Nelson-Aaen function per group
naf = NelsonAalenFitter()
ax = plt.subplot(111)
for name, grouped_df in df.groupby('group'):
    naf.fit(grouped_df["T"], event_observed=grouped_df["E"], label=name) 
    naf.plot(ax=ax)


# From datetimes to durations
from lifelines.utils import datetimes_to_durations

# start_times is a vector of datetime objects
# end_times is a vector of (possibly missing) datetime objects.
#T, E = datetimes_to_durations(start_times, end_times, freq='h')

from lifelines.utils import survival_table_from_events

table = survival_table_from_events(T, E)
print(table.head())


### Survival regression: Cox PH

from lifelines.datasets import load_rossi
rossi = load_rossi()
print(rossi.describe())
print(rossi.head())

from lifelines import CoxPHFitter
cph = CoxPHFitter()
cph.fit(rossi, 'week', 'arrest')
cph.print_summary()
cph.plot()
cph.score_

cph.plot_covariate_groups('prio', [0, 5, 10, 15])

# Now with strata
cph = CoxPHFitter()
cph.fit(rossi, 'week', event_col='arrest', strata=['race'], show_progress=True)

cph.print_summary()  # access the results using cph.summary


### Survival regression: Aalen Additive Model





