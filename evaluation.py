import importlib
import os
import sys

import numpy as np

import python.metrics as mymetrics
from python.data import Dataloader as myDataloader

if __name__ == "__main__":
    ### First part: test of the MIP model
    print("Starting Python script for evaluation")
    print("Path to data is:", sys.argv[1])
    path_to_data = sys.argv[1]
    print("Deteted available data:")
    print(os.listdir(path_to_data))

    print("Path to Repo is:", sys.argv[2])
    path_to_repo = sys.argv[2]
    sys.path.append(os.path.join(path_to_repo, "python"))
    import models

    print("MIP Model - dataset_4:")
    data_loader = myDataloader(os.path.join(path_to_data, "dataset_4"))  # Path to test dataset
    X, Y = data_loader.load(length=1000)
    
    np.random.seed(123)
    model = models.TwoClustersMIP(
        n_clusters=2, n_pieces=5
    )  # You can add your model's arguments here, the best would be set up the right ones as default.
    model.fit(X, Y)

    print(model)
    # %Pairs Explained
    pairs_explained = mymetrics.PairsExplained()
    pe_m1 = pairs_explained.from_model(model, X, Y)
    print("Percentage of explained preferences:", pe_m1)

    # %Cluster Intersection
    cluster_intersection = mymetrics.ClusterIntersection()

    Z = data_loader.get_ground_truth_labels(length=1000)
    print("% of pairs well grouped together by the model:")
    ri_m1 = cluster_intersection.from_model(model, X, Y, Z)
    print("Cluster intersection for all samples:", ri_m1)

    ### 2nd part: test of the heuristic model
    data_loader = myDataloader(os.path.join(path_to_data, "dataset_10")) # Path to test dataset
    X, Y = data_loader.load()

    indexes = np.linspace(0, len(X) - 1, num=len(X), dtype=int)
    np.random.shuffle(indexes)
    train_indexes = indexes[: int(len(indexes) * 0.8)]
    test_indexes = indexes[int(len(indexes) * 0.8) :]

    X_train = X[train_indexes]
    Y_train = Y[train_indexes]
    model = models.HeuristicModel(n_clusters=3)
    model.fit(X_train, Y_train)

    X_test = X[test_indexes]
    Y_test = Y[test_indexes]
    Z_test = data_loader.get_ground_truth_labels()[test_indexes]

    # Validation on test set
    # %Pairs Explained
    pairs_explained = mymetrics.PairsExplained()
    print("Percentage of explained preferences:", pairs_explained.from_model(model, X_test, Y_test))
    pe_m2 = pairs_explained.from_model(model, X_test, Y_test)

    # %Cluster Intersection
    cluster_intersection = mymetrics.ClusterIntersection()
    print("% of pairs well grouped together by the model:")
    print(
        "Cluster intersection for all samples:",
        cluster_intersection.from_model(model, X_test, Y_test, Z_test),
    )
    ri_m2 = cluster_intersection.from_model(model, X_test, Y_test, Z_test)

    with open(os.path.join(path_to_repo, "results.txt"), "w") as file:
        file.write(f"Model 1 Pairs explained: {pe_m1}\n")
        file.write(f"Model 1 RandIndex: {ri_m1}\n")
        file.write(f"Model 2 Pairs explained: {pe_m2}\n")
        file.write(f"Model 2 RandIndex: {ri_m2}\n")
