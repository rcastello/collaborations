import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta


def load_data(data_path):
    """
    Read csv file and copy it in a dataframe
    :param string data_path: path of the csv file containint the data
    :output Dataframe data_raw: dataset
    """
    # Import full dataset
    data_raw = pd.read_csv(data_path, header=None )
    print("Shape of raw dataset: \n", data_raw.shape)

    return data_raw

def format_data(data_raw, energy_consumption_columns):
    """
    From the input data set, select only the columns that we need for our analysis and apply
    row cleaning (wrong measurement times are deleted)
    :param Dataframe data_raw: original dataset
    :param list energy_consumption_columns: indices of columns related to total energy consumption
    :output Dataframe data: modified dataset
    """
    print("FORMATTING DATA...", end="\n")

    # Select only columns related to energy consumption
    data = data_raw.iloc[:, energy_consumption_columns]

    # Cast measurement time to datetime and set it as index
    data.loc[:, 0] = pd.to_datetime(data.loc[:, 0])
    data = data.set_index(0)

    # Remove times not multiple of 15 minutes
    data = data.loc[(data.index.minute % 15 == 0) & (data.index.second == 0)]

    # Delete repeated measurements of 2018-10-28 02:00:00 -> 2018-10-28 02:45:00
    # based on numerical index (datetime index is not unique!)
    data = data.reset_index()
    data = data.drop([28805, 28806, 28807, 28808])

    # Fill in missing values between 2018-03-25 01:45:00 -> 2018-03-25 03:00:00
    #                            and 2019-03-31 01:45:00 -> 2019-03-31 03:00:00
    # using measurements od previous week
    lines = data.loc[7976-672:7976-672+3, :]
    data = pd.concat([data.iloc[:7976, :], lines, data.iloc[7976::, :]])
    data.iloc[7976:7980, 0] = data.iloc[7976:7980, 0] + timedelta(days=7)
    lines = data.loc[43592-672:43592-672+3, :]
    data = pd.concat([data.iloc[:43592, :], lines, data.iloc[43592::, :]])
    data.iloc[43592:43596, 0] = data.iloc[43592:43596, 0] + timedelta(days=7)

    # Set back measurement time as index
    data = data.set_index(0)
    data.index.names = ['Time']

    # Dataset already starting on Monday (no drop of first rows needed)

    # Data of last 6 days are dropped, to have dataset ending on Sunday
    if data.index[-1] == pd.to_datetime('2019-11-24 01:15:00'):
        data = data.drop(data.tail(582).index.values)

    # Print dataset information
    print("Index (datetime) unique: \n", data.index.is_unique, end='\n')
    print("Shape of formatted dataset: \n", data.shape, end='\n')
    print("First measurement: \n", data.index[0], end='\n')
    print("Last measurement: \n", data.index[-1], end='\n\n')

    return data



def NaN_handling(data, NaN_number_threshold, energy_consumption_columns):
    """
    Columns with more than NaN_number_threshold NaN values are dropped,
    otherwise NaN values are replaced using a backfill.
    :param Dataframe data: dataset containing NaN
    :param list energy_consumption_columns: indices of columns related to total energy consumption of original dataset
    :param int NaN_number_threshold: buildings (columns) having more than this value of NaNs are discarded
    :output Dataframe data: modified dataset where NaNs were replaced with the immediately preceding value
    :output list selected_columns: updated list of column indices after dropping the ones containing too many NaNs
    """
    print("REMOVING NANS...", end="\n")

    # Count NaNs per column
    NaNs_per_column = data.isnull().sum(axis=0)

    # Drop columns with too many NaNs
    num_buildings = len(data.columns)
    column_name = data.columns
    selected_columns = []
    for i in range(num_buildings):
        if (NaNs_per_column.values[i] > NaN_number_threshold) | \
           (energy_consumption_columns[i+1] == 82):                   # column 82 has know inconsistencies and is dropped
            data.drop([column_name[i]], axis=1, inplace=True)
        else:
            selected_columns.append(energy_consumption_columns[i+1])  # Track non-dropped column indices (i+1 since 0 is time index)

    # Print number of dropped columns
    number_dropped = energy_consumption_columns.shape[0] - np.array(selected_columns).shape[0] - 1      # (-1 since 0 is time index)
    print("Number of dropped columns: \n", number_dropped, end="\n")
    print("Shape of dataset after NaN handling: \n", data.shape, end='\n\n')

    # Fill NaNs in non-dropped columns
    data.fillna(method='backfill',inplace=True)

    return data, selected_columns


def remove_small_buildings(data, minimum_mean_energy_consumption, original_columns):
    """
    Drop columns containing buildings with mean consuption below the threshold
    :param Dataframe data: dataset containing also small buildings
    :param int minimum_mean_energy_consumption: below this mean consumption the building (column) is dropped
    :output Dataframe data: dataset without small buildings
    """
    print("REMOVING SMALL BULIDINGS...", end="\n")

    # Drop small buildings
    num_buildings = len(data.columns)
    column_name = data.columns
    selected_columns = []
    for i in range(num_buildings):
        if data.loc[:, column_name[i]].mean() < minimum_mean_energy_consumption:
            data.drop([column_name[i]], axis=1, inplace=True)
        else:
            selected_columns.append(original_columns[i])  # Track non-dropped column indices

    # Print number of dropped columns
    number_dropped = len(original_columns) - len(selected_columns)
    print("Number of dropped columns: \n", number_dropped, end="\n")
    print("Shape of dataset after removing small buildings: \n", data.shape, end='\n\n')

    return data, selected_columns


def renumber_columns(data):
    """
    Columns are renamed as "building_" + number, where number belongs to the interval [0, len(data.columns.values)-1]
    :param Dataframe data: dataset
    :output Dataframe data: dataset with columns named in progressive order
    """
    print("RENUMBERING COLUMNS...", end="\n")
    print("Total number of buildings: \n", len(data.columns.values), end='\n')

    # Rename columns
    updated_names = []
    for i in range(len(data.columns.values)):
        updated_names.append("building_{}".format(i+1))
    data.columns = updated_names

    return data



def create_single_building_dataframe(complete_dataframe, building_number, data_per_week = 672):
    """
    From the complete dataset where every column is energy consumption of a different
    building, create a new datase containing only the energy consumption of one building
    :param Dataframe complete_dataframe: complete dataset
    :param int building_number: selected building
    :param data_per_week: number of measurement for every week (default value is 672,
                          a measurement every 15 minutes)
    :output Dataframe energy_consumptions: new dataset contaning only the selected building
    :output int total_weeks: numebr of weeks in the dataset
    """

    # Select the building for the analysis
    building_name = 'building_' + str(building_number)
    energy_consumptions = complete_dataframe[building_name]

    # Rename the measurements column
    energy_consumptions.name = 'consumption'

    # Compute total number of weeks in dataset
    total_weeks = int(energy_consumptions.shape[0]/data_per_week)
    if total_weeks % 1 != 0:
        print("WARNING: number of weeks not integer!", end='\n\n')

    # Add a column containing week number (for every measurement)
    energy_consumptions = pd.DataFrame(energy_consumptions)
    energy_consumptions['week'] = 0
    for week_number in range(0, total_weeks):
        energy_consumptions.iloc[week_number*data_per_week:(week_number+1)*data_per_week, 1] = week_number

    return energy_consumptions, total_weeks
