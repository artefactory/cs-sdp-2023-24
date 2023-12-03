from abc import abstractmethod

import numpy as np
from scipy.special import comb


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
    Rand Index metric
    https://en.wikipedia.org/wiki/Rand_index
    """

    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true):
        """main function to call the ClusterIntersection metric

        Parameters
        ----------
        y_pred (np.ndarray of shape (n_elements)):
            index (in {0, 1, ..., n}) of predicted cluster for each element
        y_true (np.ndarray of shape (n_elements)):
            index (in {0, 1, ..., n}) of ground truth cluster for each element

        Returns
        -------
        float Percentage of pairs attributed regrouped within same cluster in prediction compared to ground truth
        """
        assert y_true.shape == y_pred.shape

        truepos_plus_falsepos = comb(np.bincount(y_true), 2).sum()
        truepos_plus_falseneg = comb(np.bincount(y_pred), 2).sum()
        concatenation = np.c_[(y_true, y_pred)]
        true_positive = sum(comb(np.bincount(concatenation[concatenation[:, 0] == i, 1]), 2).sum() for i in set(y_true))
        false_positive = truepos_plus_falsepos - true_positive
        false_negative = truepos_plus_falseneg - true_positive
        true_negative = comb(len(concatenation), 2) - true_positive - false_positive - false_negative
        return (true_positive + true_negative) / (true_positive + false_positive + false_negative + true_negative)

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
