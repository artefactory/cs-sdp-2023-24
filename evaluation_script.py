import os
import sys

sys.path.append("python/")

import metrics
import numpy as np
from data import Dataloader
from models import TwoClustersMIP

if __name__ == "__main__":
    X, Y = Dataloader("path_to_test_data").load()  # Will be updated with path to other dataset

    model = (
        TwoClustersMIP()
    )  # You can add your model's arguments here, the best would be set up the right ones as default.
    model.fit(X, Y)

    # %Pairs Explained
    metric = metrics.PairsExplained()
    pairs_explained = metric(model.predict_utility(X), model.predict_utility(Y))
    print("% Pairs explained: ", pairs_explained)

    # %Cluster Intersection
    print("% of pairs well grouped together by the model:")
    print(
        metrics.ClusterIntersection()(
            np.argmax(model.predict_preference(X, Y), axis=1),
            np.argmax(GroundTruthModel.predict_preference(X, Y), axis=1),
        )
    )

    ### ADD CODE FOR THE 2ND VERSION
    ### Check the model.save and model.load
