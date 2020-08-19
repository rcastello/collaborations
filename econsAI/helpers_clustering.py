import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta


def create_weekly_matrix(energy_consumptions, data_per_week = 672, total_weeks = 98):
    """
    Creates matrix X which contains a weekly time series on each row.
    :param DataFrame energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'week' contains the week code of the measurement.
    :param data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int total_weeks: numebr of weeks in the dataset (default value is 98)
    :output matrix X: dimension total_weeks x data_per_week
    """
    total_weeks = int(energy_consumptions.shape[0]/data_per_week)
    X = np.zeros((total_weeks, data_per_week))
    rows = np.arange(data_per_week)
    for i in range(total_weeks):
        X[i, :] = energy_consumptions.iloc[i*data_per_week + rows, 0]

    return X


def compressed_week_representation(energy_consumptions, principal_components_number, data_per_week = 672, total_weeks = 98):
    """
    Computation of a compressed representation of the weeks applying PCA.
    The matrix on which PCA is applied is the dissimilarity matrix obtained
    computing euclidean distance between pairs of weekly data, represented
    in the frequency domain.

    :param DataFrame energy_consumptions:   formatted DataFrame containing measurements of energy consumption.
                                            Index is of type DatetimeIndex,
                                            column 'consumption' contains the measurement,
                                            column 'week' contains the week code of the measurement.
    :param int principal_components_number: how many singular values are used to build the compressed week
    :param data_per_week: number of measurement for every week (default value is 672, a measurement every 15 minutes)
    :param int total_weeks: numebr of weeks in the dataset (default value is 98)
    :output matrix PCA_weeks: dimension principal_components_number x total_weeks, contains per every week the first indicated                                   principal components
    :output matrix S: Singular values matrix
    """



    # Create matrix X which contains a weekly time series on each row
    X = create_weekly_matrix(energy_consumptions)

    # DFT of matrix X is computed with FFT algorithm
    FT = np.zeros((total_weeks, data_per_week), dtype=complex)
    for i in range(total_weeks):
        FT[i, :] = np.fft.fft(X[i, :])

    # Compute magnitude of the frequence spectrum
    magnitude = np.abs(FT)

    # Compute dissimilarity matrix.
    # Small M[i,j] entry means a strong similarity between the magnitude of
    # the spectra of the i-th and j-th weekly time series. Such similarity
    # is estimated with the Euclidian norm.
    # M is symmetric and diag(M)=0: only the lower triangular part is computed.
    M = np.zeros((total_weeks, total_weeks))
    for i in range(total_weeks):
        for j in range(i):
            M[i, j] = np.linalg.norm(X[i, :] - X[j, :])
            M[j, i] = M[i, j]

    # SVD decomposition of M
    U, S, Vt = np.linalg.svd(M, compute_uv=True)

    # Compute compressed representation of the weeks.
    # The higher is number of principal components chosen the less information is lost
    PCA_weeks = (U[:, 0:principal_components_number]).T @ M

    return PCA_weeks, S


def extract_clusters(clustering):
    """
    Assign each week to a cluster, otherwise indicate it as outlier
    :param output of DBSCAN clustering: results of the clustering made using DBSCAN function
    :output matrix ordered_clusters: dimension number of cluster x number of weeks, collocate every week in a cluster
    :output vector outliers: length numebr of weeks, indicate if a week is an outlier
    """
    # Assign each week to a cluster
    max_cluster_index = np.max(clustering.labels_)
    clusters = []
    for cluster_index in range(0, max_cluster_index+1):
        clusters.append(clustering.labels_ == cluster_index)
    outliers = clustering.labels_ == -1

    # Reorder clusters based on their cardinality:
    # the bigger one should contain normal weeks while the smaller ones plus
    # the outlier contain the atypical weeks
    clusters_cardinalities = []
    for i in range(0, len(clusters)):
        clusters_cardinalities.append(clusters[i].sum())
    sorted_indices = np.array(clusters_cardinalities).argsort().tolist()
    ordered_clusters = []
    for i in range(len(clusters)-1, -1, -1):
        ordered_clusters.append(clusters[sorted_indices[i]])

    return ordered_clusters, outliers


def weeks_clustering(ordered_clusters,outliers,total_weeks):
    """
    Dividing data in normal and atypical weeks:
     first cluster               -> normal weeks
     other clusters + outliers   -> atypical weeks

    :param matrix ordered_clusters: dimension number of cluster x number of weeks, collocate evry week in a cluster
    :output vector normal_weeks: indexes of normal weeks
    :output vector atypical_weeks: indexes of atypical weeks
    """


    normal_weeks = []
    atypical_weeks = []
    outliers_weeks = []
    week_indices = np.arange(0, total_weeks)
    for i in range(total_weeks):

        if (i in week_indices[ordered_clusters[0]]):
            normal_weeks.append(i)
            found_in_cluster = True
        elif(i in week_indices[outliers[0]]):
            outliers_weeks.append(i)
        else:
            atypical_weeks.append(i)


    return normal_weeks, atypical_weeks, outliers_weeks
