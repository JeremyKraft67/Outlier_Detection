# Outlier_Detection
This repository proposes a simple method to detect outliers in a time serie.
For demonstration purposes, we use the data from the online Kaggle competition:
https://www.kaggle.com/c/avazu-ctr-prediction

## How to run the project
- Download the data from Kaggle:
https://www.kaggle.com/c/avazu-ctr-prediction
- unpack the 'train.csv' into the local folder.
- Create a virtual environment with Python 3.6.9:
```
mkdir env
python3 -m venv env
source env/bin/activate
```
- Install the required libraries:
```
pip3 install -r requirements.txt
```
- launch the code:
```
python3 outlier_detection.py
```
or with your favorite IDE.

It will create 2 plots in your local folder:
- 'CTR per Hour.png' is the Click Through Ratio per 'Day-Hour'
- 'Outliers.png' is the plot with the detected outliers in red.

## Method

To detect the outliers, we use a simple moving average over the time series.
We tune the window width and consider as outliers the points lying outside the +/- 1.5 standard deviation from the moving average. 

## Discussion

This method is very simple and robust to real world data.
We check the quality of the outliers visually.
Using a window smaller than 5 is too conservative, no outliers are detected, while using a window too large results in false positives.
We find that using 6 as a window give the best results.

## Extension

We improve the algorithm by doing a Season Trend Loess decomposition to separate the time series into 3 components:
- trend
- season
- residues

We then run the outlier detection on the residue only, instead of the raw data.
We find that the STL decomposition gives interesting results but is less robust.
We get either too many false positives or it is too conservative.

We can improve the STL decomposition by using the Median Absolute Deviation instead of the standard deviation.
