
import os
from os.path import join
import sys
import numpy as np
import networkx as nx
import scipy
from scipy.stats import kurtosis, skew
import scipy.spatial.distance
from pylab import *


def read_data(dataset):
    file_path = os.getcwd() + dataset
    # Create networkx graph for all graphs in the given folder
    list_of_graphs = []
    for file in os.listdir(file_path):
        f = open(file_path + '/' + file, 'r')
        next(f)
        g = nx.Graph()
        for line in f:
            g.add_edge(int(line.split()[0]), int(line.split()[1]))
        list_of_graphs.append(g)
    return list_of_graphs

# Calculating threshold and then calculating anomalies with the help of received threshold
# Returning threshold and anomalies
def get_threshold(distance_bwn_graph):
    n = len(distance_bwn_graph)
    total = 0
    for i in range(1,n):
        difference = abs(distance_bwn_graph[i] - distance_bwn_graph[i - 1])
        total = total + difference
    median_distance = np.median(distance_bwn_graph)
    threshold = median_distance + 3 * total/(n-1)
    anomalies_detected = []
    for i in range(len(distance_bwn_graph) - 1):
        # Graph is anomalous if there are two consecutive anomalous time points in the output
        if distance_bwn_graph[i] >= threshold and distance_bwn_graph[i + 1] >= threshold:
            anomalies_detected.append(i)
    return threshold, anomalies_detected


# Get Features as defined in the paper
# This algorithm will take list of graphs as input and give output as a list of feature matrix for all nodes
def get_feature_matrix(list_of_graphs):
    feature_matrix_list = []
    
    # Repeat for every node for every graph
    for graph in list_of_graphs:
        feature_matrix = []
        for node in graph.nodes():
            # Getting degree of the node i
            degree = graph.degree(node)
            # Getting clustering coefficient of node i
            clustering_coefficient = nx.clustering(graph, node)
            
            degree_avg = 0
            # average number of node i’s two-hop away neighbors
            for neighbor in graph.neighbors(node):
                degree_avg = degree_avg + graph.degree(neighbor)
            degree_avg = degree_avg/degree

            # average clustering coefficient of neighbors of node i (N(i))
            clustering_coefficient_avg = 0
            for neighbor in graph.neighbors(node):
                clustering_coefficient_avg = clustering_coefficient_avg + nx.clustering(graph, neighbor)
            clustering_coefficient_avg = clustering_coefficient_avg/degree

            # number of edges in node i’s egonet
            egonet = nx.ego_graph(graph, node)
            edges_in_egonet = egonet.edges()
            no_of_edges_in_egonet = len(edges_in_egonet)

            # number of outgoing edges from ego(i)
            no_of_outgoing_edges_from_egonet = 0
            edges_list = set()
            for v in egonet:
                edges_list = edges_list.union(graph.edges(v))

            # Calculating the outgoing edges by eliminating edges of egonet to itself
            edges_list = list(edges_list - set(egonet.edges()))
            no_of_outgoing_edges_from_egonet = len(edges_list)

            # number of neighbors of egonet
            no_of_neighbors_of_egonet = 0
            neighbors_list = set()
            for v in egonet:
                neighbors_list = neighbors_list.union(graph.neighbors(v))

            # Calculating neighbors which are not part of the egonet itself
            neighbors_list = list(neighbors_list - set(egonet.nodes()))
            no_of_neighbors_of_egonet = len(neighbors_list)

            feature_matrix.append([degree, clustering_coefficient, degree_avg, clustering_coefficient_avg, 
                                no_of_edges_in_egonet, no_of_outgoing_edges_from_egonet, no_of_neighbors_of_egonet])
                                
        feature_matrix_list.append(feature_matrix)

    return feature_matrix_list


# Aggregator algorithm of netsimile
# This function gets input a list of feature matrix for every node for every graph and it outputs the list of signature vectors for all graphs
def aggregate(feature_matrix_list):
    list_of_sign_vector = list()
    for feature_matrix in feature_matrix_list:
        sign_vector_list = []
        # Calculate the aggregate values
        for i in range(7):
            feature_column = [node[i] for node in feature_matrix]
            med = np.median(feature_column)
            mean = np.mean(feature_column)
            std = np.std(feature_column)
            skew_val = skew(feature_column)
            kurtosis_val = kurtosis(feature_column, fisher=False)
            feature_aggregate = [med, mean, std, skew_val, kurtosis_val]

            sign_vector_list += feature_aggregate

        list_of_sign_vector.append(sign_vector_list)
    return list_of_sign_vector


# NetSimile's compare algorithm
def compare(list_of_sign_vector):
    # Canberra distance between i and i + 1
    canberra_distance = [scipy.spatial.distance.canberra(list_of_sign_vector[i+1], list_of_sign_vector[i])
            for i in range(len(list_of_sign_vector)-1)]
    return canberra_distance


# Time Series file output which has similarity between graphs
def output_time_series(distance_bwn_graph):
    timeSeriesFile = "result/" + sys.argv[1] + "_time_series.txt"
    os.makedirs(os.path.dirname(timeSeriesFile), exist_ok=True)
    f = open(timeSeriesFile, 'w+')
    for dist in distance_bwn_graph:
        f.write(str(dist) + '\n')
    f.close()


def plot_time_series_threshold(distance_bwn_graph, threshold):
    figure(figsize=(10, 5))
    plt.plot(distance_bwn_graph, "-ro")
    axhline(y=threshold, c='blue', lw=2)
    plt.title("Dataset " + sys.argv[1])
    plt.xlabel("Time Series")
    plt.ylabel("Distance (Canberra)")
    plotFile = "result/" + sys.argv[1] + "_time_series_plot.png"
    os.makedirs(os.path.dirname(plotFile), exist_ok=True)
    savefig(plotFile, bbox_inches='tight')


# NetSimile Algorithm
def net_simile(list_of_graphs):
    # This is a list of graphs provided in the dataset and we need to detect the anomaly in this dataset
    # Getting feature matrix of this graph list
    feature_matrix = get_feature_matrix(list_of_graphs)
    
    # Getting list of signature vectors from the feature matrix
    list_of_sign_vector = aggregate(feature_matrix)

    # compare the graphs for similarities and get distance between graphs
    distance_bwn_graphs = compare(list_of_sign_vector)
    
    # Threshold calculation and use threshold to detect anomalies
    threshold, anomalies_detected = get_threshold(distance_bwn_graphs)
    print(threshold)
    print("Anomalies Detected = ", anomalies_detected)

    # Generate the time series text file
    output_time_series(distance_bwn_graphs)

    # Plot the line graph of time series with a horizontal threshold bar
    plot_time_series_threshold(distance_bwn_graphs, threshold)

# Fetch the name of dataset from input argument
dataset = sys.argv[1]

# Read the dataset as a list of graphs
list_of_graphs = read_data('/datasets/'+dataset)

# Run the algorithm NetSimile
net_simile(list_of_graphs)