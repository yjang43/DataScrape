#!/usr/bin/env python
"""
The module contains functions:
    that creates vocabulary from articles under "articles" directory
    that creates BoW of each articles
    that applies PCA
    that applies HAC on PCA
"""

import numpy as np
import os
import json
import nltk
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage

PRINCIPAL_COMPONENT = 2     # set PC


def create_vocabulary(dir_path: str):
    """
    returns a list of vocabulary by collecting all the words used from articles under the
    directory specified and returns a dictionary of vocabulary with 0 for each key's value
    """
    files = os.listdir(dir_path)
    f_paths = [os.path.join(dir_path, file) for file in files]

    # create vocabulary
    vocabulary = {}
    for path in f_paths:
        with open(path) as f:
            article = json.load(f)
        tokenized_article = _clean_data(article)
        for token in tokenized_article:
            if not vocabulary.__contains__(token):
                vocabulary[token] = 0
    return vocabulary


def _clean_data(article):
    """
    tokenize article's content
    """
    text = article['text']
    return nltk.word_tokenize(text)


def create_bow(vocabulary, path):
    """
    returns normalized BoW of an article specified
    """
    # https://medium.com/@swethalakshmanan14/how-when-and-why-should-you-normalize-standardize-rescale-your-data-3f083def38ff
    # because the scale of frequency of a certain word does not solely decide the sentiment of the article,
    # BoW needs to be normalized
    # create normalized bag of words
    # create bag of words first
    bow = vocabulary.copy()
    with open(path) as f:
        article = json.load(f)
    tokenized_article = _clean_data(article)
    for token in tokenized_article:
        bow[token] += 1

    # normalization should be done here
    squared_sum = 0
    for token in bow:
        val = bow[token]
        squared_sum += val ** 2
    denominator = math.sqrt(squared_sum)
    for token in tokenized_article:
        bow[token] = bow[token] / denominator

    bow = [bow[word] for word in bow]

    return bow


def create_dataset(dir_path):
    """
    return ndarray shape of n by m, where n is the number of samples and
    m is the number of dimesions
    """
    files = os.listdir(dir_path)
    f_paths = [os.path.join(dir_path, file) for file in files]
    f_paths.sort()

    dataset = []
    vocabulary = create_vocabulary(dir_path)
    for path in f_paths:
        # print("path:", path)
        bow = create_bow(vocabulary, path)
        dataset.append(bow)

    dataset = np.array(dataset)
    print(dataset.shape)
    return dataset


def pca_transform(dataset):
    """
    pca transforms the dataset and returns X dimension dataset as specified in PRINCIPAL_COMPONENT
    """
    # input n by m ndarray,
    # where n is a number of samples and m is number of dimensions
    pca = PCA(n_components=PRINCIPAL_COMPONENT)
    pca.fit(dataset)
    data_transformed = pca.transform(dataset)
    print(data_transformed.shape)

    return data_transformed


def choose_dataset(dataset, index_list):
    """
    returns subset of the dataset chosen by the list of the indices, in a shape of m by n.
    n is the number of chosen samples and m is the number of dimensions
    """
    # because dataset is n by m matrix,
    # empty matrix will be set to 0 by m and add chosen samples
    _, dim = dataset.shape
    dataset_chosen = np.empty([0, dim])
    for index in index_list:
        dataset_chosen = np.append(dataset_chosen, [dataset[index]], axis=0)
    return dataset_chosen


def visualize_dataset(dataset):
    """
    visualizes dataset in a scatter plot
    """
    plt.figure()
    print(dataset[:, 0])
    print(dataset[:, 1])
    plt.scatter(dataset[:, 0], dataset[:, 1])
    dataset_chosen = choose_dataset(dataset, [0])
    plt.scatter(dataset_chosen[:, 0], dataset_chosen[:, 1], c='red')
    plt.show()


def hac(dataset):
    """
    scipy's linkage method wrapper.
    method is set to complete
    """
    # n by m matrix,
    # where n is the number of samples and m is the number of dimension
    return linkage(dataset, method="complete")


def classify(dataset, classification_file_path):
    """
    classify the given dataset with a pre-labeled data, which can be retrieved from "article_classification.json".
    The classification ends when a collision between labeled clusters occurs
    """

    num_of_samples = dataset.shape[0]   # n number of samples from the dataset

    with open(classification_file_path, 'r') as f:
        data_classified = json.load(f)

    # initialize information of clusters
    # in a format of:
    #   {index_a: {"representation": ..., "cluster":[...]}, index_b:...
    cluster_sets = Representation(dataset, data_classified)
    linkage_matrix = hac(dataset)

    # continue cluster until StopClustering exception
    try:
        for i in range(num_of_samples - 1):
            cluster_sets.combine(
                linkage_matrix[i][0],
                linkage_matrix[i][1],
                num_of_samples + i
            )
            print(linkage_matrix[i][0], linkage_matrix[i][1], "combined")
    except StopClustering:
        pass

    # create classification in a format of:
    #   {label_a: [...], label_b:...
    # each label contains indices of related articles to the label
    classification = {label: [] for label in data_classified}
    for label in data_classified:
        for c in cluster_sets.cluster:
            if data_classified[label].__contains__(cluster_sets.cluster[c]['representation']):
                classification[label] += cluster_sets.cluster[c]['cluster']
    print(classification)

    return classification


def visualize_classification(dataset, classification_file_path):
    """
    visualize classification with scatter plot
    """
    clusters = classify(dataset, classification_file_path)
    plt.figure()
    plt.scatter(dataset[:, 0], dataset[:, 1])

    for label in clusters:
        cluster = clusters[label]
        dataset_chosen = choose_dataset(dataset, cluster)
        print("chosen dataset are...")
        print(dataset_chosen)
        print()
        plt.scatter(dataset_chosen[:, 0], dataset_chosen[:, 1], label=label)
    plt.legend()
    plt.show()


class Representation:
    """
    data structure to contain clustering information
    """
    def __init__(self, dataset, data_classified):
        """
        initialize self.cluster as such format
        """
        num_of_samples = dataset.shape[0]
        self.data_classified = data_classified
        self.cluster = {
            index: {
                'representation': index,
                'cluster': [index]
            }
            for index in range(num_of_samples)
        }

    def combine(self, index_a, index_b, new_index):
        """combine two clusters and give the new cluster a new index"""

        def check_label(index):
            """returns label of the cluster"""
            for label in self.data_classified:
                if self.data_classified[label].__contains__(index):
                    return label
            return None

        rep_a = self.cluster[index_a]['representation']
        rep_b = self.cluster[index_b]['representation']

        if check_label(rep_a) and check_label(rep_b) and check_label(rep_a) != check_label(rep_b):
            raise StopClustering
        if check_label(rep_a):
            self.cluster[new_index] = {
                'representation': rep_a,
                'cluster': self.cluster.pop(index_a)['cluster'] + self.cluster.pop(index_b)['cluster']
            }
        else:
            self.cluster[new_index] = {
                'representation': rep_b,
                'cluster': self.cluster.pop(index_a)['cluster'] + self.cluster.pop(index_b)['cluster']
            }


class StopClustering(Exception):
    """exception to signal stop clustering"""
    pass






