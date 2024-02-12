import os

import numpy as np


class Dataloader(object):
    """Main class to load a dataset from a bunch of .npy files

    Arguments
    ----------
    path_to_data: str
        path to the directory containing the .npy files. The data in the directory is for now:
            - X.npy: (n_samples, n_features) features of the preferred elements
            - Y.npy: (n_samples, n_features) features of the unchosen elements
    """

    def __init__(self, path_to_data):
        """Data loader initialization

        Parameters
        ----------
        path_to_data: str
            path to the directory containing the .npy files. The data in the directory is for now:
        """
        self.path_to_data = path_to_data

    def load(self, length=None):
        """Main method to call to load the data

        Returns
        -------
        np.ndarray:
            Features of preferred elements
        np.ndarray:
            Features of unchosen elements
        """
        try:
            X = np.load(os.path.join(self.path_to_data, "X.npy"))
            Y = np.load(os.path.join(self.path_to_data, "Y.npy"))
        except FileNotFoundError:
            print(f"No data found at specified path {self.path_to_data}")
            return None, None

        if length is None:
            length = len(X)
        return X[:length], Y[:length]

    def get_ground_truth_labels(self, length=None):
        """Returns Ground truth cluster labels.

        Returns
        -------
        np.ndarray:
            Ground truth cluster associated to each element (x, y)
        """
        try:
            Z = np.load(os.path.join(self.path_to_data, "Z.npy"))
            if length is None:
                length = len(Z)
            return Z[:length]
        except FileNotFoundError:
            print(f"No Ground Truth labels found at specified path {self.path_to_data}")
            return None
