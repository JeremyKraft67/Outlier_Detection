#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:42:43 2020

@author: jkraft
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import time
from datetime import datetime
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose

# train_df = pd.read_csv("train.csv")

train_df_red = pd.read_csv("train_df_red.csv")
# train_df_red['hour_concat'] = train_df['hour']
# train_df.shape

# train_df_red.to_csv("train_df_red.csv", index=False)

# keep only the 2 relevant columns
# train_df_red = train_df[[ 'hour', 'click']].copy()
# check the dates
train_df_red['hour_concat'].min()
train_df_red['hour_concat'].max()



# As the data consists of 11 consecutive days in the same month,
# we can keep only the days and hours
hour_str = 14103023
def format_time(hour_str):
    """ Extract the day and hour."""
    hour_str = str(hour_str)
    day = int(hour_str[-4:-2])
    hour = int(hour_str[-2:])
    
    return day, hour

t = time.time()
time_tuple = list(map(lambda x: format_time(x), list(train_df_red['hour'])))
print("Time to extract days and hours: {}s".format(round((time.time() - t), 2)))

# train_df_red['time_stamp'] = pd.Timestamp(time_tuple)
# pd.Timestamp(time_tuple)

# extract the results
time_tuple_l = list(zip(*time_tuple))
day_l = list(time_tuple_l[0])
hour_l = list(time_tuple_l[1])
train_df_red['day'] = day_l
train_df_red['hour'] = hour_l

# def concat_day_hour(ind, day_str, hour_str):
#     day_hour = int(str(day_str[ind]) + str(hour_str[ind]))
#     return day_hour
# # create a concatenation of day and hour for the group by    
# train_df_red['day_hour'] =\
#     list(map(lambda x:
#              concat_day_hour(x,
#                  list(range(len(list(train_df_red['day'])))),
#                  list(train_df_red['day'])),
#              list(train_df_red['hour']))
#          )
    

# group by hour
ctr_df = train_df_red.groupby(['hour_concat'])['click'].agg(sum)
ctr_df = ctr_df.reset_index(drop=False)
# hour since first day
ctr_df['hour_lin'] = ctr_df.index


# plot the CTR
# plot by index to avoid blanks when plotting by 'hour_concat'
plt.scatter(list(ctr_df['hour_lin']), ctr_df['click'])
plt.title('CTR per hour',
          fontsize=14, fontweight='bold')
plt.ylabel('Quantity of clicks', fontweight='bold', fontsize=12)
plt.xlabel('Day & Hour', fontweight='bold', fontsize=12)
# the data corresponds to 10 consecutive days


# plot the CTR
ctr_df_hour = train_df_red.groupby(['day', 'hour'])['click'].agg(sum)
ctr_df_hour = ctr_df_hour.reset_index(drop=False)

groups = ctr_df_hour.groupby("day")
for name, group in groups:
    plt.plot(group["hour"], group["click"], marker="o", linestyle="", label=name)

plt.legend()
plt.title('Clicks per hour',
          fontsize=14, fontweight='bold')
plt.ylabel('Quantity of clicks', fontweight='bold', fontsize=12)
plt.xlabel('Hour', fontweight='bold', fontsize=12)
plt.legend(loc='upper right')


# =============================================================================
# outliers detection
# =============================================================================

def plot_outliers(ctr_df, window, moving_avg=True):
    
    if moving_avg:
        # plot the ouliers
        subplot_num = int(str(math.ceil(len(window_list)/qty_sublpot_col))
                          + str(qty_sublpot_col)
                          + str(plot_ind)) + 1                        
        plt.subplot(subplot_num)
    
    ctr_df_outlier = ctr_df[ctr_df['is_outlier']].copy()
    ctr_df_no_outlier = ctr_df[~ctr_df['is_outlier']].copy()
    plt.scatter(ctr_df_outlier['hour_lin'], ctr_df_outlier['click'],
                c='red', s=4, label='outlier')
    plt.scatter(ctr_df_no_outlier['hour_lin'], ctr_df_no_outlier['click'],
            c='green', s=4, label='no outlier')

    plt.plot('hour_lin', 'mean',
              data=ctr_df, linewidth=1)
    plt.title('Outliers for window {win}'.format(win=window),
              fontsize=fontsize, fontweight='bold')
    plt.ylabel('Quantity of clicks', fontweight='bold', fontsize=fontsize)
    plt.xlabel('Hour', fontweight='bold', fontsize=fontsize)

    # plt.legend(bbox_to_anchor=(1, 1), loc='upper left', prop=fontP)
    plt.legend(loc='upper right',prop=fontP)
    

# find the outliers using the moving average method
# compute the rolling mean, then plot several windows to select the best one

# plot many windows
window_list = list(range(3, 12))
thresh = 1.5
qty_sublpot_col = 2

plt.figure()
fontsize = 8
fontP = FontProperties()
fontP.set_size('xx-small')


for plot_ind, window in enumerate(window_list):
    # compute rolling means
    roll_avg = ctr_df['click'].rolling(window=window, center=True)
    
    roll_avg.mean()
    roll_avg.std()
    ctr_df['mean'] = roll_avg.mean()
    ctr_df['std'] = roll_avg.std()
    
    # compute outliers
    ctr_df['is_outlier'] =  (ctr_df['click'] >= (ctr_df['mean'] + thresh * ctr_df['std']))\
                            | (ctr_df['click'] <= (ctr_df['mean'] - thresh * ctr_df['std']))
    
    plot_outliers(ctr_df, window, True)
plt.subplots_adjust(hspace=0.8)


# Based on a visual inspection, a window of 6 seems to work well
plt.close()
plot_outliers(ctr_df, window=6, moving_avg=False)


# compute outliers with STL

# plot the results of the STL decomposition

window = 5 # only used to plot the moving average
thresh = 1.5
stl = STL(ctr_df['click'], period=24, low_pass=25,
          seasonal=15,
          seasonal_deg=0,
          trend_deg=1,
          robust=True)
residue = stl.fit()
fig1 = res.plot()

# extract residues
residue = res.resid
ctr_df['residue'] = residue
# compute rolling means
roll_avg = ctr_df['click'].rolling(window=window, center=True)
mean = ctr_df['residue'].mean()
std = ctr_df['residue'].std()

# compute outliers
ctr_df['is_outlier'] =  (ctr_df['residue'] >= (mean + thresh * std))\
                      | (ctr_df['residue'] <= (mean - thresh * std))
plt.close()                     
plot_outliers(ctr_df, window, False)  
  
# There seems to be a lot of false positive.
# Let's tune the threshold.
       
# Threshold tuning for the STL
thresh_list = [1.5, 2, 2.5, 3, 3.5, 4]        
qty_sublpot_col = 2

plt.figure()
fontsize = 8
fontP = FontProperties()
fontP.set_size('xx-small')

for plot_ind, thresh in enumerate(thresh_list):

    stl = STL(ctr_df['click'], period=24, low_pass=25,
              seasonal=15,
              seasonal_deg=0,
              trend_deg=1,
              robust=True)
    residue = stl.fit()
    # fig1 = res.plot()
    
    # extract residues
    residue = res.resid
    ctr_df['residue'] = residue
    # compute rolling means
    roll_avg = ctr_df['click'].rolling(window=window, center=True)
    mean = ctr_df['residue'].mean()
    std = ctr_df['residue'].std()
    
    # compute outliers
    ctr_df['is_outlier'] =  (ctr_df['residue'] >= (mean + thresh * std))\
                          | (ctr_df['residue'] <= (mean - thresh * std))
    plot_outliers(ctr_df, thresh, True)
plt.subplots_adjust(hspace=0.8)

# A threshold of 3 might work.
# But we have either too many false positive or the STL is too conservative.


# Let's try the Median Absolute Deviation from the median


thresh_list = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]        
qty_sublpot_col = 2

plt.figure()
fontsize = 8
fontP = FontProperties()
fontP.set_size('xx-small')

for plot_ind, thresh in enumerate(thresh_list):

    stl = STL(ctr_df['click'], period=24, low_pass=25,
              seasonal=15,
              seasonal_deg=0,
              trend_deg=1,
              robust=True)
    residue = stl.fit()
    # fig1 = res.plot()
    
    # extract residues
    residue = res.resid
    ctr_df['residue'] = residue
    # compute rolling means
    roll_avg = ctr_df['click'].rolling(window=window, center=True)
    
    median = ctr_df['residue'].median()
    mad = abs(median - ctr_df['residue']).median()
    ctr_df['is_outlier'] =  abs(ctr_df['residue'])\
                              >= (median + thresh * mad) 
    plot_outliers(ctr_df, thresh, True)
plt.subplots_adjust(hspace=0.8)

# A large threshold of 5 seems acceptable.
