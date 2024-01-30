import math
import pickle
from abc import abstractmethod

import gurobipy as grb
import numpy as np


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)  # Weights cluster 1
        weights_2 = np.random.rand(num_features)  # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0])  # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1])  # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1)  # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, *, enable_two_clusters_optimization=True):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        enable_two_clusters_optimization: bool, default=True
            If true and `n_clusters` is 2 then use an optimized formulation of the problem.
            Otherwise, use a formulation that is valid for any value of `n_clusters`.
        """
        self.seed = 123
        self.model = self.instantiate(
            L=n_pieces,
            K=n_clusters,
            enable_two_clusters_optimization=enable_two_clusters_optimization,
        )

    def instantiate(self, L, K, *, enable_two_clusters_optimization=True):
        """Instantiation of the MIP Variables - To be completed."""
        model = grb.Model("Market Segmentation")
        self.L = L
        self.K = K
        self.M = 2  # upper bound for our problem
        self.two_clusters_optimization = (K == 2) and enable_two_clusters_optimization
        return model

    def compute_xl(self, min, max, l):
        return min + l * (max - min) / self.L

    def compute_score(self, k, instance):
        scores = []
        for i, x in enumerate(instance):
            # score is equal to max score
            if x == self.criteria_maxs[i]:
                scores.append(self.u[k, i, self.L])
                continue

            # score is below min score or above max score (for new instances)
            if x < self.criteria_mins[i]:
                scores.append(0)
                continue
            if x > self.criteria_maxs[i]:
                scores.append(self.u[k, i, self.L])
                continue

            # other scores
            l = math.floor(
                self.L
                * (x - self.criteria_mins[i])
                / (self.criteria_maxs[i] - self.criteria_mins[i])
            )
            x_l = self.compute_xl(self.criteria_mins[i], self.criteria_maxs[i], l)
            x_l_plus_1 = self.compute_xl(
                self.criteria_mins[i], self.criteria_maxs[i], l + 1
            )
            delta_x = x - x_l
            slope = (self.u[k, i, l + 1] - self.u[k, i, l]) / (x_l_plus_1 - x_l)
            scores.append(self.u[k, i, l] + slope * delta_x)

        return grb.quicksum(scores)

    def store_result(self):
        self.utility_functions = np.zeros(
            (self.K, self.n, self.L + 1), dtype=np.float32
        )
        for k in range(self.K):
            for i in range(self.n):
                for l in range(1, self.L + 1):
                    self.utility_functions[k, i, l] = self.u[k, i, l].x

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        # To be completed
        self.P = X.shape[0]
        self.n = X.shape[1]
        self.EPS = 10 ** (-3)

        # Decision variables
        if self.two_clusters_optimization:
            self.delta = {
                j: self.model.addVar(name=f"delta_{j}", vtype=grb.GRB.BINARY)
                for j in range(self.P)
            }
        else:
            self.delta = {
                (k, j): self.model.addVar(name=f"delta_{k}_{j}", vtype=grb.GRB.BINARY)
                for j in range(self.P)
                for k in range(self.K)
            }

        self.u = {
            (k, i, l): self.model.addVar(name=f"u_{k}_{i}_{l}", lb=0) if l else 0
            for k in range(self.K)
            for i in range(self.n)
            for l in range(self.L + 1)
        }
        self.sigma = {j: self.model.addVar(name=f"s_{j}", lb=0) for j in range(self.P)}

        all_elements = np.concatenate([X, Y], axis=0)
        self.criteria_mins = all_elements.min(axis=0)
        self.criteria_maxs = all_elements.max(axis=0)

        # Constraints

        # C1 - Each pair is explained by at least one cluster
        if not self.two_clusters_optimization:
            for j in range(self.P):
                self.model.addConstr(
                    grb.quicksum([self.delta[k, j] for k in range(self.K)]) >= 1
                )

        # C2 - The utility function for each cluster is modeled by a sum of piecewise linear functions being equal to 1
        for k in range(self.K):
            self.model.addConstr(
                grb.quicksum([self.u[k, i, self.L] for i in range(self.n)]) == 1
            )

        # C3 - Each criterion is modeled by an increasing piecewise linear function
        for k in range(self.K):
            for i in range(self.n):
                for l in range(self.L):
                    self.model.addConstr(self.u[k, i, l + 1] - self.u[k, i, l] >= 0)

        # C4 - delta_k_j implication
        if self.two_clusters_optimization:
            for j in range(self.P):
                self.model.addConstr(
                    self.compute_score(0, X[j])
                    + self.sigma[j]
                    - self.compute_score(0, Y[j])
                    >= self.EPS + self.M * (self.delta[j] - 1)
                )
                self.model.addConstr(
                    self.compute_score(1, X[j])
                    + self.sigma[j]
                    - self.compute_score(1, Y[j])
                    >= self.EPS - self.M * self.delta[j]
                )
        else:
            for k in range(self.K):
                for j in range(self.P):
                    self.model.addConstr(
                        self.compute_score(k, X[j])
                        + self.sigma[j]
                        - self.compute_score(k, Y[j])
                        >= self.EPS + self.M * (self.delta[(k, j)] - 1)
                    )

        # Objective
        self.obj = grb.quicksum(list(self.sigma.values()))
        self.model.setObjective(self.obj, grb.GRB.MINIMIZE)

        self.model.params.outputflag = 0

        self.model.update()

        self.model.optimize()

        assert self.model.status != grb.GRB.INFEASIBLE

        self.store_result()

        return self.model

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        utilities = []
        for k in range(self.K):
            utilities.append(np.array([self.compute_score(k, x).getValue() for x in X]))
        return np.stack(utilities, axis=1)


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model."""
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
