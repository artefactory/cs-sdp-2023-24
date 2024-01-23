import os
import sys

sys.path.append("python/")

import metrics
import numpy as np
from data import Dataloader
from models import HeuristicModel, TwoClustersMIP

if __name__ == "__main__":  
    print("Starting Python script for evaluation")
    print("Path to data is:", sys.argv[1])
    path_to_data = sys.argv[1]
    print(os.listdir(path_to_data))

    print("MIP Model - dataset_4:")
    ### First part: test of the MIP model
    data_loader = Dataloader(os.path.join(path_to_data, "dataset_4"))  # Path to test dataset

    X, Y = data_loader.load()

    model = TwoClustersMIP(
        n_clusters=2, n_pieces=5
    )  # You can add your model's arguments here, the best would be set up the right ones as default.
    model.fit(X, Y)

    # %Pairs Explained
    pairs_explained = metrics.PairsExplained()
    print("Percentage of explained preferences:", pairs_explained.from_model(model, X, Y))

    # %Cluster Intersection
    cluster_intersection = metrics.ClusterIntersection()

    Z = data_loader.get_ground_truth_labels()
    print("% of pairs well grouped together by the model:")
    print("Cluster intersection for all samples:", cluster_intersection.from_model(model, X, Y, Z))

    print("Heuristic Model - dataset_10:")
    ### 2nd part: test of the heuristic model
    data_loader = Dataloader(os.path.join(path_to_data, "dataset_10"))  # Path to test dataset
    X, Y = data_loader.load()

    np.random.seed(123)
    indexes = np.linspace(0, len(X) - 1, num=len(X), dtype=int)
    np.random.shuffle(indexes)
    train_indexes = indexes[: int(len(indexes) * 0.8)]
    test_indexes = indexes[int(len(indexes) * 0.8) :]

    X_train = X[train_indexes]
    Y_train = Y[train_indexes]
    model = HeuristicModel(n_clusters=3)
    model.fit(X_train, Y_train)

    X_test = X[test_indexes]
    Y_test = Y[test_indexes]
    Z_test = data_loader.get_ground_truth_labels()[test_indexes]

    # Validation on test set
    # %Pairs Explained
    pairs_explained = metrics.PairsExplained()
    print("Percentage of explained preferences:", pairs_explained.from_model(model, X_test, Y_test))

    # %Cluster Intersection
    cluster_intersection = metrics.ClusterIntersection()
    print("% of pairs well grouped together by the model:")
    print(
        "Cluster intersection for all samples:",
        cluster_intersection.from_model(model, X_test, Y_test, Z_test),
    )
