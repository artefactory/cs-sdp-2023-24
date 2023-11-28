from abc import abstractmethod

import numpy as np
from sklearn.metrics import r2_score


class BaseMetric:
    """Base Class for metrics. Should be inherited by all metrics"""

    def __init__(self):
        """Initialization of the metric, can be used to setup different behaviours of the metric"""
        super().__init__()

    @abstractmethod
    def __call__(self, X, Y):
        """Method where the metric should be defined

        Parameters
        ----------
        X: np.ndarray
            Role can change depending on the metric
        Y: np.ndarray
            Role can change depending on the metric
        """
        return


class PairsExplained(BaseMetric):
    """
    Computes the percentage of pairs that are explained by at least a cluster
    """

    def __init__(self):
        super().__init__()

    def __call__(self, Ux, Uy):
        """main function to call the PairsExplained metric

        Parameters
        ----------
        Ux (np.ndarray of shape (n_elements, n_clusters)):
            utilities of preferred elements for each cluster
        Uy (np.ndarray of shape (n_elements, n_clusters)):
            utilities of non preferred elements for each cluster

        Ux and Uy are organised such that \forall i, x_i > y_i and we compare Ux[i] with Uy[i] for each cluster

        Returns
        -------
        float
            percentage of pairs explained by at least a cluster
        """
        assert Ux.shape == Uy.shape
        if len(Ux.shape) == 1:
            Ux = np.expand_dims(Ux, axis=-1)
            Uy = np.expand_dims(Uy, axis=-1)
        return np.sum(np.sum(Ux - Uy > 0, axis=1) > 0) / len(Ux)

    def from_model(self, model, X, Y):
        """Method to use the metric from a model and data.

        Parameters
        ----------
        model : BaseModel
            Model to be evaluated
        X : np.ndarray
            data to be evaluated: features of preferred elements
        Y : np.ndarray
            data to be evaluated: features of unchosen elements

        Returns
        -------
        float
            percentage of pairs explained by at least a cluster
        """
        Ux = model.predict_utility(X)
        Uy = model.predict_utility(Y)
        return self(Ux, Uy)


class ClusterIntersection(BaseMetric):
    """
    Computes the average intersection percentage of predicted clusters vs ground truth.
    For now, only works for two clusters
    """

    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true):
        """main function to call the ClusterIntersection metric

        Parameters
        ----------
        y_pred (np.ndarray of shape (n_elements)):
            index (in {0, 1}) of predicted cluster for each element
        y_true (np.ndarray of shape (n_elements)):
            index (in {0, 1}) of ground truth cluster for each element

        Returns
        -------
        float Percentage of pairs attributed regrouped within same cluster in prediction compared to ground truth
        """
        assert y_true.shape == y_pred.shape
        return np.max([np.sum(y_true == y_pred), np.sum(y_true == 1 - y_pred)]) / y_pred.shape[0]

    def from_model(self, model, X, Y, y_true):
        """Method to use the metric from a model and data.

        Parameters
        ----------
        model : BaseModel
            Model to be evaluated
        X : np.ndarray
            data to be evaluated: features of preferred elements
        Y : np.ndarray
            data to be evaluated: features of unchosen elements
        y_true : np.ndarray
            Ground truth cluster associated to each element (x, y)

        Returns
        -------
        float
            float Percentage of pairs attributed regrouped within same cluster in prediction compared to ground truth
        """
        y_pred = model.predict_cluster(X, Y)
        return self(y_pred, y_true)


class CommonPreferences(BaseMetric):
    """Computes the percentage of common preferences between two clusters from pairs of utilities.
    Metric used two compare two clusters (Ground Truth vs Predicted or not)
    """

    def __init__(self, from_utility=True, num_features=None):
        self.from_utility = from_utility
        self.num_features = num_features
        if not self.from_utility:
            assert self.num_features is not None

    def __call__(self, y_pred, y_true):  # keep utilities in tuples to get only two inputs ?
        """ """
        if self.from_utility:
            Ux_1, Uy_1 = y_pred
            Ux_2, Uy_2 = y_true
            assert Ux_1.shape == Ux_2.shape == Uy_1.shape == Uy_2.shape

            # Compute preferences of each model
            preferences_1 = np.argmax(np.stack([Ux_1, Uy_1], axis=1), axis=1)
            preferences_2 = np.argmax(np.stack([Ux_2, Uy_2], axis=1), axis=1)

            # Percentage of common preferences
            return np.sum(preferences_1 == preferences_2) / len(preferences_1)

        else:  # from model
            # How to chose the x, y to be compared ?
            # How to know how many features the model handles ?
            A = []
            B = []

            # Creation of the pairs to be compared
            for i in range(10):
                a = np.zeros(10)
                a[i] = 1
                for j in range(10):
                    if i != j:
                        b = np.zeros(10)
                        b[j] = 1
                        A.extend([a, a * 0.5, a * 0.5, a])
                        B.extend([b, b * 0.5, b, b * 0.5])

            A = np.stack(A)
            B = np.stack(B)

            raise NotImplemented
