# TODO: are all this imports needed?

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import DBSCAN

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

from copy import deepcopy

from helpers_preprocessing import *
from helpers_clustering import *
from helpers_plot import *
from helpers_modelweek import *
from helpers_linear_regression import *
from helpers_postprocessing import * 


def analyze_building(data_clean, building_number, plots = True):
    """
    Given the already preprocessed dataset and a chosen building number, this function
    compute the complete analysis for the corresponding time series.

    :param DataFrame data_clean: original preprocessed dataset
    :param int building_number: index of the chosen buildings
    :param bool plots: True if needed to visualize all the plots during the Analysis
    :output DataFrame df_output: DataFrame summarizing the results of anomaly detection
                                    Index is the index of the week,
                                    column 'Atypical days number': number of atypical days in every weeks,
                                    column ' Anomalies numebr': number of anomalies number not contained in atypical days.
    :output double error: relative error computed on non atypical days
    """


    data_per_day = 96
    data_per_week = data_per_day*7
    energy_consumptions, total_weeks = create_single_building_dataframe(data_clean, building_number, data_per_week)
    energy_consumptions.head(5)

    first_singular_value_multiplier = 0.012
    principal_components_number = 2

    # FT + PCA on selected building
    PCA_weeks, S = compressed_week_representation(energy_consumptions, principal_components_number, data_per_week, total_weeks)

    # Run DBSCAN to identify the clusters in the space defined by the Principal Components
    epsilon = S[0]*first_singular_value_multiplier
    clustering = DBSCAN(eps=epsilon, min_samples=9).fit(PCA_weeks.T)

    # Assign each week to a cluster and reorder clusters based on their cardinality
    # (the bigger one should contain normal weeks and will be plotted in blue; the remaining clusters and outliers (black)
    # are the atypical weeks
    ordered_clusters, outliers = extract_clusters(clustering)
    
    if (plots):
        # Scatterplot the result of the clustering
        plot_clusters(clustering, ordered_clusters, PCA_weeks, outliers, building_number)

        # Plot data in groups of 4 weeks using different colors, corresponding to the clusters:
        # blue     -> normal weeks
        # yellow   -> remaining clusters
        # black    -> outliers


        plot_clustered_weeks(total_weeks, data_per_week, ordered_clusters, energy_consumptions)

    # Dividing data in normal and atypical weeks:
    # first cluster     -> normal weeks
    # other clusters + outliers   -> atypical
    normal_weeks, atypical_weeks= weeks_clustering(ordered_clusters, total_weeks)

    # Using only normal weeks, creation of the model week 
    model_week = weekly_model(energy_consumptions, normal_weeks)
    
    
    if (plots):
        plot_model_week(model_week, data_per_week, energy_consumptions)

    regularized_energy_consumptions,df_atypical_weeks = regularize_data(energy_consumptions, atypical_weeks, model_week)

    lags =  np.array([96*7-1, 96*7, 96*7+1, 96*7*2-1, 96*7*2, 96*7*2+1, 96*7*3-1, 96*7*3, 96*7*3+1, 96*7*4-1, 96*7*4])
    data_lm = lm_dataframe_categorical_features(regularized_energy_consumptions, lags)



    y = data_lm.y
    X = data_lm.drop(['y'], axis=1)

    # reserve 30% of data for testing
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    # Linear regression model 
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    if (plots):
        plot_coefficients_lr(lr, X_train)

    results_linear_regression = steepness_anomaly_proof_regression(lr, regularized_energy_consumptions,energy_consumptions, data_lm, model_week,lags, total_weeks)

    if (plots):
        plot_anomalies(results_linear_regression)
    
    df_output = compute_output(results_linear_regression, df_atypical_weeks)
    error= compute_error(results_linear_regression)
    
    return df_output, error



def analyze_building_feature_selection(data_clean, building_number):

    data_per_day = 96
    data_per_week = data_per_day*7
    energy_consumptions, total_weeks = create_single_building_dataframe(data_clean, building_number, data_per_week)
    energy_consumptions.head(5)

    first_singular_value_multiplier = 0.012
    principal_components_number = 2

    # FT + PCA on selected building
    PCA_weeks, S = compressed_week_representation(energy_consumptions, principal_components_number, data_per_week, total_weeks)

    # Run DBSCAN to identify the clusters in the space defined by the Principal Components
    epsilon = S[0]*first_singular_value_multiplier
    clustering = DBSCAN(eps=epsilon, min_samples=9).fit(PCA_weeks.T)

    # Assign each week to a cluster and reorder clusters based on their cardinality
    # (the bigger one should contain normal weeks and will be plotted in blue; the remaining clusters and outliers (black)
    # are the atypical weeks
    ordered_clusters, outliers = extract_clusters(clustering)

    # Dividing data in normal and atypical weeks:
    # first cluster     -> normal weeks
    # other clusters + outliers   -> atypical
    normal_weeks, atypical_weeks= weeks_clustering(ordered_clusters, total_weeks)

    # Using only normal weeks, creation of the model week 
    model_week = weekly_model(energy_consumptions, normal_weeks)
    

    regularized_energy_consumptions,df_atypical_weeks = regularize_data(energy_consumptions, atypical_weeks, model_week)

    lags = np.array([ 96*7-1,96*7,96*7+1, 96*7*2-1,96*7*2,96*7*2+1, 96*7*3-1,96*7*3,96*7*3+1, 96*7*4-1, 96*7*4])
    data_lm = lm_dataframe_all_features(regularized_energy_consumptions, lags)

    y = data_lm.y
    X = data_lm.drop(['y'], axis=1)

    # reserve 30% of data for testing
    X_train, X_test, y_train, y_test = timeseries_train_test_split(X, y, test_size=0.3)

    first=1
    for depth in range(16,25):
        for max_feat in range(11,16):
            regr = RandomForestRegressor(n_estimators=200,criterion="mse",max_depth=depth,max_features=max_feat)
            regr.fit(X_train, y_train)
            if(first):
                feat_im = pd.DataFrame(regr.feature_importances_,X_train.columns)
                feat_im=feat_im.rename(columns={0: "build_{}_{}_{}".format(building_number,depth,max_feat)})
                first = 0
            else:
                feat_im["build_{}_{}_{}".format(building_number,depth,max_feat)] =regr.feature_importances_
     
    return feat_im, X_train



                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                       


