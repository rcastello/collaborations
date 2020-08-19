import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta


def format_energy_plot(y_min, y_max):
    """
    Utility to format all time series plots
    :param double y_min: minimum y values displayed
    :param double y_max: maximum y values displayed
    """
    plt.gca().set_ylim([y_min, 1.2*y_max])  # TODO: fix y_max to add also legend
    plt.xticks(rotation=20)
    plt.gca().set_ylabel('Energy consumption [kWh]')
    plt.grid('True')
    
    
def plot_clusters(clustering, ordered_clusters, PCA_weeks, outliers, building_number=None):
    """
    Plot clusters,every point is a week: normal weeks are in blue, outliers are in black
    :param output of DBSCAN clustering: results of the clustering made using DBSCAN function
    :param matrix ordered_clusters: dimension number of cluster x number of weeks, collocate evry week in a cluster
    :param matrix PCA_weeks: dimension principal_components_number x total_weeks, contains per every week the first indicated                                   principal components
    :param vector outliers: length numebr of weeks, indicate if a week is an outlier
    :param int building_number: number of the considered building
    """
    max_cluster_index = np.max(clustering.labels_)
    
    # Scatterplot the result
    for cluster_index in range(0, max_cluster_index+1):
        plt.scatter(PCA_weeks[0, ordered_clusters[cluster_index]], PCA_weeks[1, ordered_clusters[cluster_index]])
    plt.scatter(PCA_weeks[0, outliers], PCA_weeks[1, outliers], color='black')
    if building_number != None:
        plt.title('Building ' + str(building_number))
    plt.gca().set_xlabel('PC1')
    plt.gca().set_ylabel('PC2')
    plt.show()
    
    
def plot_clustered_weeks(total_weeks, data_per_week, ordered_clusters, energy_consumptions):
    """
    Plot time serie in groups of 4 weeks using different colors to distinghuish between the clustered weeks:
     blue             -> normal weeks
     yellow, black    -> atypical weeks
    :param int total_weeks: numebr of weeks in the dataset (default value is 98)
    :param data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param matrix ordered_clusters: dimension number of cluster x number of weeks, collocate evry week in a cluster
    :param DataFrame energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                            Index is of type DatetimeIndex, 
                                            column 'consumption' contains the measurement,
                                            column 'week' contains the week code of the measurement.
    """

    
    data_per_group = data_per_week*4
    group_range = [0, int(total_weeks/4)]
    week_indices = np.arange(0, total_weeks)
    
    for group_index in range(group_range[0], group_range[1]):
        plt.figure(figsize=(20, 4))
        for local_week_index in range(0, 4):
            global_week_index = group_index*4 + local_week_index
            # Look for the current week in every cluster (plot with relative color)
            found_in_cluster = False
            for cluster_index in range(0, len(ordered_clusters)):
                if (global_week_index in week_indices[ordered_clusters[cluster_index]]):
                    week_color = 'C' + str(cluster_index)
                    energy_consumptions.loc[energy_consumptions['week'] == global_week_index, 'consumption']\
                                       .plot(grid='True',rot=45, color=week_color)
                    found_in_cluster = True
            # If week not found then it's outlier (plot in black)
            if not found_in_cluster:
                energy_consumptions.loc[energy_consumptions['week'] == global_week_index, 'consumption']\
                                   .plot(grid='True', rot=45, color='black')
        plt.title('Group ' + str(group_index))
        format_energy_plot(energy_consumptions.consumption.min(), 
                           energy_consumptions.consumption.max())
        plt.show()
        

        
def plot_model_week(model_week, data_per_week, energy_consumptions):
    """
    Plot the model of a "standard" week, computed using the median value for every measurement in th week
    :param vector model_week: model
    :param data_per_week: number of measurement for every week 
    :param DataFrame energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                            Index is of type DatetimeIndex, 
                                            column 'consumption' contains the measurement,
                                            column 'week' contains the week code of the measurement.
    """
    points = np.arange(data_per_week)
    plt.plot(points, model_week, color='C1')
    format_energy_plot(energy_consumptions.consumption.min(), 
                       energy_consumptions.consumption.max())
    plt.locator_params(axis='x', nbins=7)
    plt.title("Model week")
    plt.show()
    

def plot_regularized_weeks(regularized_energy_consumptions, atypical_weeks, total_weeks, data_per_week = 672):
    """
    Plot the weeks that have been regularized, i.e. the days with a large number of anomalies have been substituted with the           corresponging day of themodel week 
   :param DataFrame regularized_energy_consumptions:   regularized DataFrame containing measurements of energy consumption.
                                                       Index is of type DatetimeIndex, 
                                                       column 'consumption' contains the measurement,
                                                       column 'week' contains the week code of the measurement.
    :param vector atypical_weeks: indexes of atypical weeks 
    :param int total_weeks: numebr of weeks in the dataset 
    :param data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    """

    rows = np.arange(data_per_week)
    for i, global_week in enumerate(atypical_weeks):
        regularized_energy_consumptions.iloc[rows+global_week*data_per_week, 0]\
                           .plot(grid='True',rot=45)
        plt.title('Week ' + str(atypical_weeks[i]))
        format_energy_plot(regularized_energy_consumptions.consumption.min(), 
                           regularized_energy_consumptions.consumption.max())
        plt.show()
        
        
def plot_coefficients_lr(model, X_train):
    """
    Plots sorted coefficient values of the model
    :param model: linear regression model
    :param matrix X_train: data used for training of the linear regression model
    """
    
    coefs = pd.DataFrame(model.coef_, X_train.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed')
    

def plot_anomalies(regression_dataframe, groups_number = 24,save_fig = False):
    """
    Plot time series in group of 4 weeks, real values are in blue, model is orange and red points are the detected anomalies 
    :param DataFrame regression_dataframe: DataFrame containing measurements of energy consumption.
                                           Index is of type DatetimeIndex, 
                                           column 'consumption' contains the real measurement
                                           column 'model' contains the linear regression model
                                           column 'anomalies' contains a bool indicating if the point is anomalous
    """
    
    # Plot data in groups of 4 weeks
    # number of weekly measurements:    672
    # number of full available weeks:   total_weeks

    data_per_group = 672*4
    group_range = [0, groups_number]
    rows = np.arange(data_per_group)
    for i in range(group_range[0], group_range[1]):
        plt.figure(figsize=(20, 4))
        ts_for_plot = regression_dataframe.iloc[rows+i*data_per_group, 0]
        plt.plot(pd.to_datetime(ts_for_plot.index), ts_for_plot.values)
        ts_for_plot = regression_dataframe.iloc[rows+i*data_per_group, 1]
        plt.plot(pd.to_datetime(ts_for_plot.index), ts_for_plot.values)
        anomalies_bool = regression_dataframe.iloc[rows+i*data_per_group, 2].values

        anomalies_time = regression_dataframe.index[i*data_per_group + np.where(anomalies_bool == True)[0]]
        anomalies_value = regression_dataframe.iloc[i*data_per_group +np.where(anomalies_bool == True)[0], 0].values
        plt.plot(pd.to_datetime(anomalies_time), anomalies_value, 'or')

        plt.title('Group ' + str(i))
        format_energy_plot(regression_dataframe.consumption.min(), 
                           regression_dataframe.consumption.max())
        if (save_fig):
            plt.savefig('Anomalies' + str(i) +'.png')
       
    plt.show()
    

def plot_weeks(results_linear_regression, weeks_number, first_week_index, figure_size=(20, 4), image_name='', data_per_week=672):
    """
    Plot time series of a weeks_number weeks, real values are in blue, model is orange and red points are the detected anomalies 
    :param DataFrame regression_dataframe: DataFrame containing measurements of energy consumption.
                                           Index is of type DatetimeIndex, 
                                           column 'consumption' contains the real measurement
                                           column 'model' contains the linear regression model
                                           column 'anomalies' contains a bool indicating if the point is anomalous
    :param int weeks_number: how many weeks to plot
    :param list first_week_index: index of the first weeks to plot (warning: no check on last weeks!)
    :param tuple figure_size: speccifies the size of the plot
    :param string image_name: name of the file in which the image is saved (image not saved if no name provided)
    :param data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    """
    
    # Set figure size
    plt.figure(figsize=figure_size)
    
    # Build the vector containing measurements indices
    rows = np.arange(data_per_week)
    time_indices = rows + first_week_index*data_per_week
    for week_index in range(first_week_index + 1, first_week_index + weeks_number):
        time_indices = np.hstack((time_indices, rows + week_index*data_per_week))
    
    # Plot both real values and model
    for column_index in range(2):
        ts_for_plot = results_linear_regression.iloc[time_indices, column_index]
        plt.plot(pd.to_datetime(ts_for_plot.index), ts_for_plot.values)
    
    # Plot anomalies
    anomalies_bool = results_linear_regression.iloc[time_indices, 2].values
    anomalies_time = results_linear_regression.index[first_week_index*data_per_week + np.where(anomalies_bool == True)[0]]
    anomalies_value = results_linear_regression.iloc[first_week_index*data_per_week +np.where(anomalies_bool == True)[0], 0].values
    plt.plot(pd.to_datetime(anomalies_time), anomalies_value, 'or')

    format_energy_plot(results_linear_regression.consumption.min(), 
                       results_linear_regression.consumption.max())
    plt.legend(['Measurements', 'Model', 'Anomalies'], loc='upper right')
    if image_name != '':
        plt.savefig(image_name + '.png')
    plt.show()
    
  