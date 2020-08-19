import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from copy import deepcopy
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

from helpers_clustering import create_weekly_matrix


def regularize_data(energy_consumptions, atypical_weeks, model_week, data_per_week = 672, data_per_day = 96, total_weeks = 98):
    """
    Creation of a new dataset "regularized_energy_consumptions" where the atypical days that charcaterize the atypical
    weeks found using the clustering, are replace with the corresponding days of the model week.
    Moreover the dataset "df_atypical_weeks", containing the number of atypical
    days for each week, is assembled

    :param DataFrame energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'week' contains the week code of the measurement.
    :param vector atypical_weeks: indexes of the weeks that are classified as atypical by the clustering phase
    :param vector model_week: model week computed using the median of the normal weeks
    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every week (default value is 96, a measurement every 15 minutes)
    :param int total_weeks: number of total weeks in the dataset (default value is 98)
    :output DataFrame regularized_energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                                         Index is of type DatetimeIndex,
                                                         column 'consumption' contains the measurement,
                                                         column 'week' contains the week code of the measurement,
                                                         column 'atypical day' contains a bool indicating if that measurament
                                                                was replaced with the model one.
    :output DataFrame df_atypical_weeks: DataFrame containing information about
                                        the number of detected atypical days.
                                        Index is of type int and contains the week number,
                                        column 'Atypical days number' contains the number of atypical
                                                days found in that week.

    """

    regularized_energy_consumptions = deepcopy(energy_consumptions)
    regularized_energy_consumptions['atypical day'] = False

    # day_model is a list of 7 arrays containing the model for every day of the week
    day_model = []
    for i in range(0, 7):
        day_model.append(model_week[i*data_per_day:(i+1)*data_per_day])

    # X = (num weeks, num data per week)
    X = create_weekly_matrix(energy_consumptions)
    X_atypical = X[atypical_weeks,:]


    # Create a dataframe containing the atypical weeks and indicating how many atypical days there are in each one
    df_atypical_weeks = pd.DataFrame({'Atypical days number' :np.zeros(total_weeks)})
    df_atypical_weeks.index.names = ['Week number']

    # Computing the relative error of every measurement with respect to the model week
    error_anomalous = np.abs(X_atypical - model_week)
    relative_error = error_anomalous/model_week
    num_atypical_weeks = np.shape(X_atypical)[0]
    columns_day = np.arange(data_per_day)

    # Counting the number of anomalies in each day of the atypcial weeks:
    # a measurement is anomalous if the relative error is greater than 30%
    anomalies_perday = np.zeros((num_atypical_weeks, 7))
    for week_index in range(num_atypical_weeks):
        for day_index in range(7):
            anomalies_perday[week_index, day_index] = (relative_error[week_index, day_index*data_per_day+columns_day] > 0.3).sum()

    # Indacting for each day of the week, in which week nuber it has been
    # detected as anomalous.
    # A day is anomalous if the number of anomalies in that day is
    # greater than 30, meaning that more than 31% of the measurement in that day
    # are anomalous

    days = []
    for day_index in range(7):
        current_day_anoumalous_weeks = []
        for week_index in range(num_atypical_weeks):

            if anomalies_perday[week_index, day_index] > 30:
                current_day_anoumalous_weeks.append(atypical_weeks[week_index])
        days.append(current_day_anoumalous_weeks)

    # For each week, count how many atypical days are detected
    for week_index in atypical_weeks:
        count = 0
        for i in range(7):

            # Weeks that has atypical days on days i
            weeks_i = np.asarray(days[i])
            count += np.count_nonzero(weeks_i == week_index)
        df_atypical_weeks.iloc[week_index, 0] = count

    # In the dataset regularized_energy_consumptions, replace the detected
    # atypical days measurements with the corresponding values of the model week.
    rows = np.arange(data_per_day)
    for day_index, day in enumerate(days):
        for global_week in day:
            regularized_energy_consumptions.iloc[rows + data_per_day*day_index + global_week*data_per_week, 0] = day_model[day_index]
            regularized_energy_consumptions.iloc[rows + data_per_day*day_index + global_week*data_per_week, 2] = True

    return regularized_energy_consumptions, df_atypical_weeks

def lm_dataframe_all_features(regularized_energy_consumptions, lags, data_per_week = 672, data_per_day = 96, total_weeks = 98):
    """
    Creating a copy of the initial dataframe that will be used to construct the linear regression model.
    Lags are used as features, as well as some categorical ones. This function
    consider all the posssible features, without prevoius feature selection.

    :param DataFrame regularized_energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                                       Index is of type DatetimeIndex,
                                                       column 'consumption' contains the measurement,
                                                       column 'week' contains the week code of the measurement.
                                                       column 'atypical day' contains a bool indicating if that measurament
                                                              was replaced with the model one.
    :param vector lags: indicates which previous measurament are used as features for the regression model
    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every week (default value is 96, a measurement every 15 minutes)
    :param int total_weeks: number of total weeks in the dataset (default value is 98)
    :output DataFrame data_lm: DataFrame containing target and features that will be
                                used for linear regression.
                                Index is of type DatetimeIndex,
                                column 'y' contains the targets, i.e. the value that
                                the time series ssume in each point
                                the remaining column contains different typed of features.
    """
    # Creating a copy of the initial dataframe and use it to construct the linear regression model
    data_lm = pd.DataFrame(regularized_energy_consumptions.consumption.copy())
    data_lm.columns = ["y"]


    # Add previous lags indicated in "lags"
    data_lm = add_lags(data_lm, lags)

    # Categorical features:

    # in which day of the week
    data_lm = add_categorical_day(data_lm)
    # if the measurement is in the central part of a day
    data_lm = add_categorical_central_part(data_lm)
    # if the measurement is in the central part of a working day or in a sunday
    data_lm = add_categorical_workday(data_lm)
    # in which month of the year
    data_lm = add_categorical_month(data_lm)

    return data_lm


def lm_dataframe_categorical_features(regularized_energy_consumptions, lags, data_per_week = 672, data_per_day = 96, total_weeks = 98):
    """
    Creating a copy of the initial dataframe that will be used to construct the linear regression model.
    Lags are used as features, as well as some categorical ones. This function
    consider only the feature selected by the feature selection process.

    :param DataFrame regularized_energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                                       Index is of type DatetimeIndex,
                                                       column 'consumption' contains the measurement,
                                                       column 'week' contains the week code of the measurement.
                                                       column 'atypical day' contains a bool indicating if that measurament
                                                              was replaced with the model one.
    :param vector lags: indicates which previous measurament are used as features for the regression model
    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every week (default value is 96, a measurement every 15 minutes)
    :param int total_weeks: number of total weeks in the dataset (default value is 98)
    :output DataFrame data_lm: DataFrame containing target and features that will be
                                used for linear regression.
                                Index is of type DatetimeIndex,
                                column 'y' contains the targets, i.e. the value that
                                the time series ssume in each point
                                the remaining column contains different typed of features.
    """
    # Creating a copy of the initial dataframe and use it to construct the linear regression model

    data_lm = pd.DataFrame(regularized_energy_consumptions.consumption.copy())
    data_lm.columns = ["y"]

    # Add previous lags indicated in "lags"
    data_lm = add_lags(data_lm, lags)

    # Categorical features:

    # if the measurement is in the central part of a working day or in a sunday
    data_lm = add_categorical_workday(data_lm)
    # if the measurement is in the month from may to august
    data_lm = add_categorical_months_summer(data_lm)

    return data_lm


def steepness_anomaly_proof_regression(regression_model, regularized_energy_consumptions, energy_consumptions, data_lm, model_week, lags,total_weeks,data_per_week = 672):

    """
    Construction of the regression model for the whole time series.
    Using the parameters optimized by the training phase of linear regression model, the prediction is made for every point, if         this prediction is far from the real value, the point is classified as anomalous. When the prediction is done, if a previous       point that has to be used as lag is anomalous, the predicted value is considered and not the real one.
    The way in which the anomaly check is computed takes care of in which part of the day the measurement is taken.
    :param model regression_model: model given by the linear regression training
    :param DataFrame energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'week' contains the week code of the measurement.
    :param DataFrame data_lm: DataFrame containing in the first column the real measurements and in the other columns the fetures                                used for regression
    :param vector model_week: model week computed using the median of the normal weeks
    :param vector lags: indicates which previous measurament are used as features for the regression model
    :param int total_weeks: number of weeks in the dataset
    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :output DataFrame hardcoded: DataFrame containing the constructed model
                                 Index is of type DatetimeIndex,
                                 column 'consumption' contains the measurement,
                                 column 'model' contains the prediction made by the model
                                 column 'anomaly' contains a bool indicating if the point is considered as anomalous
    """
    # new dataset used for forecasting and anomaly detection
    hardcoded_regression = pd.DataFrame(energy_consumptions.consumption)
    hardcoded_regression['model'] = np.hstack((np.tile(model_week, 4), np.zeros(data_per_week*(total_weeks - 4))))
    hardcoded_regression['anomaly'] = False
    hardcoded_regression['atypical day'] = regularized_energy_consumptions['atypical day']


    # "threshold" is used to detect anomalies in the central parts of the day, between 8a.m. and 7p.m.
    # Anomalies in the "head and tail" of each day are detected with a different criterion, based on relative error
    # that depends on "threshold_steep"
    # This double standard proved to be very convenient, since the "head and tail" section of each day is very steep.
    threshold = compute_anomaly_threshold_central_part_of_day(data_lm)
    threshold_steep = 0.8


    # First 4 weeks of the time series cannot be predicted by linear regression model since
    # the prevoius lags are not available: model week is replaced instead of the model
    real_values =  hardcoded_regression['consumption'].iloc[0:data_per_week*4]
    model_values = hardcoded_regression['model']

    # anomaly detection for first 4 weeks
    for row_index in range(data_per_week*4):
        time_of_the_day = (row_index%672)%96
        hardcoded_regression = detect_anomaly(threshold, threshold_steep, time_of_the_day, row_index, hardcoded_regression,model_values[row_index], real_values[row_index])



    # Step by step prediction avoiding to use anomalous values:
    # starting from 5th week until the end of the dataset

    for row_index in range(4*data_per_week, total_weeks*data_per_week):
        data_point = np.zeros(data_lm.columns.values.shape[0] - 1)
        features_mask_bool = hardcoded_regression.iloc[row_index-lags, 2].values
        features_mask_columns = np.zeros(features_mask_bool.shape[0])

        # Check if the lags used as features to predict the next point are anomalous,
        # if that is the case, the previous predicted model values are taken
        # as features for that measuraments
        for i in range(0, features_mask_bool.shape[0]):
            if features_mask_bool[i] == True:
                data_point[i] = hardcoded_regression.iloc[row_index - lags[i], 1]
            if features_mask_bool[i] == False:
                data_point[i] = hardcoded_regression.iloc[row_index - lags[i], 0]
        # Prediction using the trained linear model, saving it in hardcoded_regression dataset
        prediction = regression_model.predict(data_point.reshape(1, -1))
        hardcoded_regression.iloc[row_index, 1] = prediction


        # Detect if the predicted point is anomalous, evaluating it in different way
        # respect to the time of the day in which the measurement is taken
        time_of_the_day=(row_index%672)%96
        hardcoded_regression = detect_anomaly(threshold, threshold_steep, time_of_the_day, row_index, hardcoded_regression,prediction, hardcoded_regression.iloc[row_index,0])

    return hardcoded_regression


def detect_anomaly(threshold, threshold_steep, time_of_the_day, row_index, hardcoded_regression, prediction, real):

    """
    Detection of anomalies in real values of time series, using two different
    criterion for central part of the day and remaining part.

    :param double threshold : used for the central part
    :param double threshold_steep : used for begin and end of the day
    :param int time_of_day: indicate in which part of the day the measurement is taken
    :param int row_index: row index of the time series corresponding to the measurement to analyse
    :param,output DataFrame hardcoded_regression: DataFrame containing the constructed model
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'model' contains the prediction made by the model
                                            column 'anomaly' contains a bool indicating if the point is considered as anomalous
                                            column 'atypical day' contains a bool indicating if that measurament
                                                    is part of a day detected as atypical in the first phase of the analysis.
    :param double prediction: value predicted by the linear model
    :param double real: actual value of the time series
    :output vector data_point: regularized features vector for the considered point
    """
    absolute = np.abs(prediction - real)
    relative = absolute/prediction

    if (time_of_the_day > -1 and time_of_the_day < 32) or (time_of_the_day > 76):
        if relative > threshold_steep:
            hardcoded_regression.iloc[row_index, 2] = True
    elif absolute > threshold:
        hardcoded_regression.iloc[row_index, 2] = True

    return hardcoded_regression


def compute_anomaly_threshold_central_part_of_day(data_lm,total_weeks=98,data_per_week = 672):
    """
    This function computes autmoatically a threshold for each building. Such threshold is then used to detect if measurements
    in the central part of the day (opening hours basically) are anomalous or not. For each opening day, the std of the
    measurements during the opening hours is computed. This is done for each day between Monday and Saturday of the data.
    The computed std values are then averaged.
    For the "head and tail" of each day, a different threshold is used, set in the function steepness_anomaly_proof_regression.
    :param DataFrame data_lm: DataFrame containing target and features that will be
                                used for linear regression.
                                Index is of type DatetimeIndex,
                                column 'y' contains the targets, i.e. the value that
                                the time series ssume in each point
                                the remaining column contains different typed of features.
    :param int total_weeks: number of weeks in the dataset (default value is 98)
    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every day (default value is 96, a measurement every 15 minutes)
    :output vector data_point: regularized features vector for the considered point
    """

    std_vector = np.zeros(total_weeks*6)
    central_measurements_of_day = []

    for row_index in range(0, (total_weeks - 4)*data_per_week):
        if(row_index%672 == 0):
            #this matrix contains the measurements of the central part of the days from Monday to Saturday of the week currently             #being analysed. It's initialized to zeros at the start of each new week.
            current_week_central_measurements_of_days = np.zeros((45,6))
        day_index = (row_index//96)%7
        if (day_index != 6):
            time_of_the_day = (row_index%672)%96
            #If the time is between 8 a.m. and 7 p.m., the measurements are kept in a vector. Their std is then computed.
            if (time_of_the_day >= 32 and time_of_the_day <= 76):
                current_week_central_measurements_of_days[time_of_the_day - 32,day_index]=data_lm.iloc[row_index,0]
        if (row_index%671 == 0):
            std_days_current_week = np.std(current_week_central_measurements_of_days, axis = 0)
            week_index = row_index//672
            std_vector[week_index*6:(week_index+1)*6] = std_days_current_week
    threshold = np.mean(std_vector)
    return 1.96*threshold



def add_lags(data_lm, lags):
    """
    Add the indicated lags as features of the target variable,
    contained in the y column of data_lm

    :param vector lags: indicates which previous measurament are used as features for the regression model
    :param,output DataFrame data_lm: DataFrame containing target and features that will be
                                    used for linear regression.
                                    Index is of type DatetimeIndex,
                                    column 'y' contains the targets, i.e. the value that
                                    the time series ssume in each point
                                    the remaining column contains different typed of features.
    """



    for i in lags:
        data_lm["lag_{}".format(i)] = data_lm.y.shift(i)
    data_lm = data_lm.iloc[max(lags):]
    return data_lm


def add_categorical_day(data_lm, data_per_week = 672, data_per_day = 96):
    """
    Categorical features representing the day of the week are added

    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every day (default value is 96, a measurement every 15 minutes)
    :param,output DataFrame data_lm: DataFrame containing target and features that will be
                                    used for linear regression.
                                    Index is of type DatetimeIndex,
                                    column 'y' contains the targets, i.e. the value that
                                    the time series ssume in each point
                                    the remaining column contains different typed of features.
    """



    categorical_day = np.zeros((data_lm.shape[0],7))
    for i in range(categorical_day.shape[0]):
        categorical_day[i,(i%data_per_week)//data_per_day] = 1
    for i in range(7):
        data_lm["day_{}".format(i+1)] = categorical_day[:,i]

    return data_lm

def add_categorical_central_part(data_lm, data_per_week = 672, data_per_day = 96):
    """
    Categorical features describing if the measurement is in the central part of
    the day are added

    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every day (default value is 96, a measurement every 15 minutes)
    :param,output DataFrame data_lm: DataFrame containing target and features that will be
                                        used for linear regression.
                                        Index is of type DatetimeIndex,
                                        column 'y' contains the targets, i.e. the value that
                                        the time series ssume in each point
                                        the remaining column contains different typed of features.
    """

    categorical_day_central = np.zeros((data_lm.shape[0],7))
    for i in range(categorical_day_central.shape[0]):
        time_of_day=(i%data_per_week)%data_per_day
        if(time_of_day>28 and time_of_day<80):
            categorical_day_central[i,(i%data_per_week)//data_per_day] = 1

    for i in range(7):
        data_lm["central_part_day_{}".format(i+1)] = categorical_day_central[:,i]

    return data_lm

def add_categorical_workday(data_lm, data_per_week = 672, data_per_day = 96):
    """
    Categorical features describing if the measuremente is in the central
    part of a working day or in a Sunday

    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every day (default value is 96, a measurement every 15 minutes)
    :param,output DataFrame data_lm: DataFrame containing target and features that will be
                                    used for linear regression.
                                    Index is of type DatetimeIndex,
                                    column 'y' contains the targets, i.e. the value that
                                    the time series ssume in each point
                                    the remaining column contains different typed of features.
    """
    categorical_working_day_central = np.zeros(data_lm.shape[0])
    for i in range(categorical_working_day_central.shape[0]):
        time_of_day=(i%data_per_week)%data_per_day
        day_of_week=(i//data_per_day)%7

        if(time_of_day>28 and time_of_day<80):
            if(day_of_week != 6):
                categorical_working_day_central[i] = 1
    data_lm["central_part_working_day"] = categorical_working_day_central

    return data_lm

def add_categorical_month(data_lm, data_per_week = 672, data_per_day = 96):
    """
    Categorical features representing the month of the year are added

    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every day (default value is 96, a measurement every 15 minutes)
    :param,output DataFrame data_lm: DataFrame containing target and features that will be
                                        used for linear regression.
                                        Index is of type DatetimeIndex,
                                        column 'y' contains the targets, i.e. the value that
                                        the time series ssume in each point
                                        the remaining column contains different typed of features.
    """
    categorical_months = np.zeros((data_lm.shape[0],12))
    for i in range(data_lm.shape[0]):
        month=((i//2688)+1)%12
        time_of_day=(i%data_per_week)%data_per_day
        if(time_of_day>28 and time_of_day<80):
            if(month == 0):
                categorical_months[i,0] = 1
            else:
                categorical_months[i,month] = 1

    # Add only categorical month from May to August
    for i in range(1, 12):
        data_lm["month_{}".format(i+1)] = categorical_months[:,i]
    return data_lm

def add_categorical_months_summer(data_lm, data_per_week = 672, data_per_day = 96):
    """
    Categorical features representing the month of the year are added only
    for the months from May to August

    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int data_per_day: number of measurement for every day (default value is 96, a measurement every 15 minutes)
    :param,output DataFrame data_lm: DataFrame containing target and features that will be
                                        used for linear regression.
                                        Index is of type DatetimeIndex,
                                        column 'y' contains the targets, i.e. the value that
                                        the time series ssume in each point
                                        the remaining column contains different typed of features.
    """
    categorical_months = np.zeros((data_lm.shape[0],12))
    for i in range(data_lm.shape[0]):
        month=((i//2688)+1)%12
        time_of_day=(i%data_per_week)%data_per_day
        if(time_of_day>28 and time_of_day<80):
            if(month == 0):
                categorical_months[i,0] = 1
            else:
                categorical_months[i,month] = 1

    # Add only categorical month from May to August
    for i in range(5, 9):
        data_lm["month_{}".format(i+1)] = categorical_months[:,i]
    return data_lm


def timeseries_train_test_split(X, y, test_size):
    """
    Perform train-test split with respect to time series structure
    param: matrix X: data regression matrix
    param: vector y: targets
    param: int test_size: percentage of the data used for testing
    """

    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))

    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]

    return X_train, X_test, y_train, y_test
