#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 23:20:36 2020

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

def format_time(hour_str):
    """ Extract the day and hour."""
    hour_str = str(hour_str)
    day = int(hour_str[-4:-2])
    hour = int(hour_str[-2:])
    
    return day, hour

def concat_day_hour(ind, day_str, hour_str):
    day_hour = str(day_str[ind]) + '.' + str(hour_str[ind])
    return day_hour

def plot_outliers(ctr_df, window, multiplot=True, window_name='window',
                  subplot_num=None):
    """
    Plot the outliers

    Parameters
    ---------
    ctr_df : Pandas dataframe
        Dataframe with at least the columns: ['click', 'hour_lin', 'is_outlier']
    window : int
        Window of the moving average
    multiplot : bool
       True for multiplot, False otherwise
   window_name : str
       Either 'window' for the moving average or 'threshold' for the STL.
   subplot_num : int
       If multiplot, the quantity of plots.

       
    Returns
    -------
    plt : Matplotlib plot
        The plot.

    """
    
    fontsize = 8
    fontP = FontProperties()
    fontP.set_size('xx-small')
    
    if multiplot:
        plt.subplot(subplot_num)
    
    # avoid bugs when working with strings as x lables
    ctr_df['hour_lin'] = ctr_df.index
    
    ctr_df_outlier = ctr_df[ctr_df['is_outlier']].copy()
    ctr_df_no_outlier = ctr_df[~ctr_df['is_outlier']].copy()
    plt.scatter(ctr_df_outlier['hour_lin'], ctr_df_outlier['click'],
                c='red', s=4, label='outlier')
    plt.scatter(ctr_df_no_outlier['hour_lin'], ctr_df_no_outlier['click'],
            c='green', s=4, label='no outlier')

    plt.plot('hour_lin', 'mean',
              data=ctr_df, linewidth=1)
    plt.title('Outliers for ' + window_name + ' {win}'.format(win=window),
              fontsize=fontsize, fontweight='bold')
    plt.ylabel('Quantity of clicks', fontweight='bold', fontsize=fontsize)
    plt.xlabel('Hour', fontweight='bold',
               fontsize=fontsize)
    plt.xticks(rotation='vertical')
    plt.grid()
    plt.legend(loc='upper right',prop=fontP)
    
    return plt
    