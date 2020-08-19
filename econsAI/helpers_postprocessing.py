import numpy as np
from copy import deepcopy


def compute_output(regression_result, df_atypical_weeks, data_per_day=96, data_per_week=672, total_weeks=98):
    """
    Create a DataFrame summarizing the results.
    For every week (row) count atypical days and number of anomalies in non-atypical days.

    :param DataFrame regression_result: DataFrame containing the constructed model
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'model' contains the prediction made by the model
                                            column 'anomaly' contains a bool indicating if the point is considered as anomalous
                                            column 'atypical day' contains a bool indicating if that measurament
                                                    is part of a day detected as atypical in the first phase of the analysis.
    :param DataFrame df_atypical_weeks: contains numbers of atypical days for every week
    :param int data_per_day: number of measurement for every day (default value is 96, a measurement every 15 minutes)
    :param int data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int total_weeks: number of weeks in the dataset (default value is 98)
    :output DataFrame final_table: DataFrame summarizing the results of anomaly detection
                                    Index is the index of the building,
                                    column 'Relative percentage error' contains the realtive error computed on non atypical days,
                                    column 'Percentage of anomalies' contains the percentage of pointwise anaomalies
                                            with respect to all the measurements non contained in atypical days
                                    column 'Percentage of atypical days ' contains the percentage of atypical days with respect to the total
    """

    final_table = deepcopy(df_atypical_weeks)
    final_table['Anomalies number'] = 0

    for row_index in range(regression_result.shape[0]):

        # Compute week index (in [0, total_weeks-1])
        week_index = np.floor(row_index / data_per_week)

        # Skip atypical days from anomalies count
        if regression_result.loc[regression_result.index[row_index], 'atypical day'] == True:
            continue

        # Update counter when anomaly is found
        elif regression_result.loc[regression_result.index[row_index], 'anomaly'] == True:
            final_table.loc[week_index, 'Anomalies number'] += 1

    return final_table




def compute_error(regression_dataframe):
    """
    Return the error of the model:mean of the relative error (real_value - predicted_value)/real_value on all the
    measurements that are not in atypical days

    :param DataFrame regression_result: DataFrame containing the constructed model
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'model' contains the prediction made by the model
                                            column 'anomaly' contains a bool indicating if the point is considered as anomalous
                                            column 'atypical day' contains a bool indicating if that measurament
                                                    is part of a day detected as atypical in the first phase of the analysis.
    :output double error_real: percentage relative error
    """
    atypical_bool = regression_dataframe.iloc[:, 3].values
    normal_values = regression_dataframe.iloc[np.where(atypical_bool == False)[0],0].values
    normal_prediction = regression_dataframe.iloc[np.where(atypical_bool == False)[0], 1].values

    indexes = np.where(normal_values == 0)[0]

    relative_error_real = np.abs(normal_values - normal_prediction)/normal_values
    relative_error_real[indexes] = np.abs(normal_values[indexes] - normal_prediction[indexes])/normal_prediction[indexes]

    error_real = np.mean(relative_error_real)

    return error_real*100
