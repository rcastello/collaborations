import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

from helpers_clustering import create_weekly_matrix

def weekly_model(energy_consumptions, normal_weeks):
    """
    Create a model week using the median value for every measure point in the week (672 points), using only the weeks
    detected as normal by the clustering phase.
    :param DataFrame energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'week' contains the week code of the measurement.
    :param vector normal_weeks: indexes of normal weeks
    :output vector model_week: resulting model (length 672)
    """

    # X = (num weeks, num data per week)
    X = create_weekly_matrix(energy_consumptions)
    # Extract only normal weeks
    X_model = X[normal_weeks, :]

    # Compute the median for every measurement of the week (672 points) using t
    # normal weeks
    model_week = np.median(X_model, axis = 0)

    return model_week
