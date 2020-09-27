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
import logging

import outlier_detection_helper_functions as od_hf

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(funcName)s:\n %(message)s')
logging.Formatter(fmt='%(asctime)s')
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S', level=logging.INFO)

# =============================================================================
# # Load the data
# =============================================================================

# open the raw data located in the same folder as the script
train_df = pd.read_csv("train.csv")
# Let's examine the data
train_df.dtypes
subset_df = train_df.iloc[:1000,:]
subset_df
logging.info("Finished to load the raw data")

# keep only the 2 relevant columns
train_df_red = train_df[[ 'hour', 'click']].copy()
# save the smaller dataset
train_df_red.to_csv("train_df_red.csv", index=False)

# load the smaller dataset for speed
train_df_red = pd.read_csv("train_df_red.csv")

# check the dates
train_df_red['hour'].min()
train_df_red['hour'].max()
logging.info("Saved the data with the relevant columns as 'train_df_red.csv'.")

# =============================================================================
# Data preparation
# =============================================================================
    
# group by hour
ctr_df = train_df_red.groupby(['hour'])['click'].agg(sum)
ctr_df = ctr_df.reset_index(drop=False)

# As the data consists of 10 consecutive days in the same month,
# we can keep only the days and hours

t = time.time()
time_tuple = list(map(lambda x: od_hf.format_time(x), list(ctr_df['hour'])))
print("Time to extract days and hours: {}ms".format(round((time.time() - t)*1000, 2)))

# extract the results
time_tuple_l = list(zip(*time_tuple))
day_l = list(time_tuple_l[0])
hour_l = list(time_tuple_l[1])
ctr_df['day'] = day_l
ctr_df['hour'] = hour_l

# create a concatenation of day and hour for prettier x abisses in the plots 
ctr_df['hour_lin'] =\
    list(
        map(lambda x:
          od_hf.concat_day_hour(x,
              list(ctr_df['day']),
              list(ctr_df['hour'])),
              list(range(len(list(ctr_df['day']))))
              )
          )


# plot the CTR
# plot by index to avoid blanks when plotting by 'hour_concat'

plt.plot(ctr_df['hour_lin'], ctr_df['click'],
         linewidth=1,markersize=2,
          marker="o")
plt.title('CTR per hour',
          fontsize=14, fontweight='bold')
plt.ylabel('Quantity of clicks', fontweight='bold', fontsize=12)
plt.xlabel('Day & Hour', fontweight='bold', fontsize=12)
plt.xticks(np.arange(0, 240, 8.0))
plt.xticks(rotation='vertical')
plt.grid()
# the data corresponds to 10 consecutive days
title = 'CTR per Hour.png'
plt.savefig(title)
logging.info("Figure: '" + title + "' saved.")

# plot the CTR per hour
ctr_df_hour = ctr_df.groupby(['day', 'hour'])['click'].agg(sum)
ctr_df_hour = ctr_df_hour.reset_index(drop=False)

plt.close()
groups = ctr_df_hour.groupby("day")
for name, group in groups:
    plt.plot(group["hour"], group["click"],
             linewidth=1,markersize=2,
              marker="o",label=name)

plt.legend()
plt.title('Clicks per hour',
          fontsize=14, fontweight='bold')
plt.ylabel('Quantity of clicks', fontweight='bold', fontsize=12)
plt.xlabel('Hour', fontweight='bold', fontsize=12)
plt.xticks(np.arange(0, 24, 1.0))
plt.legend(title='Day', loc='upper right')

logging.info("Finished preparing the data.")

# =============================================================================
# outliers detection
# =============================================================================


# find the outliers using the moving average method
# compute the rolling mean, then plot several windows to select the best one

# plot many windows
window_list = list(range(3, 12))
thresh = 1.5
qty_sublpot_col = 2

plt.close()
plt.figure()

for plot_ind, window in enumerate(window_list):
    
    # plot the ouliers
    subplot_num = int(str(math.ceil(len(window_list)/qty_sublpot_col))
                  + str(qty_sublpot_col)
                  + str(plot_ind)) + 1                        
    
    # compute rolling means
    roll_avg = ctr_df['click'].rolling(window=window, center=True)
    
    roll_avg.mean()
    roll_avg.std()
    ctr_df['mean'] = roll_avg.mean()
    ctr_df['std'] = roll_avg.std()
    
    # compute outliers
    ctr_df['is_outlier'] =  (ctr_df['click'] >= (ctr_df['mean'] + thresh * ctr_df['std']))\
                            | (ctr_df['click'] <= (ctr_df['mean'] - thresh * ctr_df['std']))
    
    od_hf.plot_outliers(ctr_df, window, True,
                        window_name='window',
                        subplot_num=subplot_num)
plt.subplots_adjust(hspace=0.8)


# Based on a visual inspection, a window of 6 seems to work well
plt.close()
od_hf.plot_outliers(ctr_df, window=6, multiplot=False)
title = 'Outliers.png'
plt.savefig(title)
logging.info("Figure: '" + title + "' saved.")

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
fig1 = residue.plot()

# extract residues
residue = residue.resid
ctr_df['residue'] = residue
# compute rolling means
roll_avg = ctr_df['click'].rolling(window=window, center=True)
mean = ctr_df['residue'].mean()
std = ctr_df['residue'].std()

# compute outliers
ctr_df['is_outlier'] =  (ctr_df['residue'] >= (mean + thresh * std))\
                      | (ctr_df['residue'] <= (mean - thresh * std))
plt.close()                     
od_hf.plot_outliers(ctr_df, thresh, False, window_name='threshold')  
  
# There seems to be a lot of false positive.
# Let's tune the threshold.
       
# Threshold tuning for the STL
thresh_list = [1.5, 2, 2.5, 3, 3.5, 4]        
qty_sublpot_col = 2

plt.close()
plt.figure()

# plot the ouliers

for plot_ind, thresh in enumerate(thresh_list):
    
    subplot_num = int(str(math.ceil(len(thresh_list)/qty_sublpot_col))
                  + str(qty_sublpot_col)
                  + str(plot_ind)) + 1  

    stl = STL(ctr_df['click'], period=24, low_pass=25,
              seasonal=15,
              seasonal_deg=0,
              trend_deg=1,
              robust=True)
    residue = stl.fit()
    # fig1 = res.plot()
    
    # extract residues
    residue = residue.resid
    ctr_df['residue'] = residue
    # compute rolling means
    roll_avg = ctr_df['click'].rolling(window=window, center=True)
    mean = ctr_df['residue'].mean()
    std = ctr_df['residue'].std()
    
    # compute outliers
    ctr_df['is_outlier'] =  (ctr_df['residue'] >= (mean + thresh * std))\
                          | (ctr_df['residue'] <= (mean - thresh * std))
    od_hf.plot_outliers(ctr_df, thresh, True,
                        window_name='threshold',
                        subplot_num=subplot_num)
plt.subplots_adjust(hspace=0.8)

# A threshold of 3 might work.
# But we have either too many false positive or the STL is too conservative.


# Let's try the Median Absolute Deviation from the median

thresh_list = [1.5, 2, 2.5, 3, 4, 5, 6, 7, 8]        
qty_sublpot_col = 2

plt.figure()


for plot_ind, thresh in enumerate(thresh_list):
    
    subplot_num = int(str(math.ceil(len(thresh_list)/qty_sublpot_col))
                  + str(qty_sublpot_col)
                  + str(plot_ind)) + 1  

    stl = STL(ctr_df['click'], period=24, low_pass=25,
              seasonal=15,
              seasonal_deg=0,
              trend_deg=1,
              robust=True)
    residue = stl.fit()
    # fig1 = res.plot()
    
    # extract residues
    residue = residue.resid
    ctr_df['residue'] = residue
    # compute rolling means
    roll_avg = ctr_df['click'].rolling(window=window, center=True)
    
    median = ctr_df['residue'].median()
    mad = abs(median - ctr_df['residue']).median()
    ctr_df['is_outlier'] =  abs(ctr_df['residue'])\
                              >= (median + thresh * mad) 
    od_hf.plot_outliers(ctr_df, thresh, True,
                        window_name='threshold',
                        subplot_num=subplot_num)
plt.subplots_adjust(hspace=0.8)

# A large threshold of 5 seems acceptable.

logging.info("Finished analyzing the outliers.")