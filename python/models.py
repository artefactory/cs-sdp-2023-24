import math
import pickle
from abc import abstractmethod

import metrics
import numpy as np
from gurobipy import *
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from python.uta import PairwiseUTA


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
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

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
        """
        return np.stack(
            [np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1
        )


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, n_criteria, precision=0.0001):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.K = n_clusters
        self.L = n_pieces
        self.n = n_criteria
        self.epsilon = precision
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        model = Model("Simple PL modelling")
        self.Xs = [model.addVar(name=f"x_{i}") for i in range(self.n)]
        self.Us = [
            [
                [model.addVar(name=f"u_{k}_{i}_{l}") for l in range(self.L + 1)]
                for i in range(self.n)
            ]
            for k in range(self.K)
        ]
        # Function must be non-decreasing
        for k, i, l in zip(range(self.K), range(self.n), range(self.L - 1)):
            model.addConstr(self.Us[k][i][l] <= self.Us[k][i][l + 1])
        return model

    def u_k_i(self, k, i, X, values: bool = False):
        x = X[i]
        if x >= self.maxs[i]:
            return self.Us[k][i][-1] if not values else self.Us[k][i][-1].X
        l = math.floor(self.L * (x - self.mins[i]) / (self.maxs[i] - self.mins[i]))
        x_l = self.mins[i] + l * (self.maxs[i] - self.mins[i]) / self.L
        width = x - x_l

        ukil = self.Us[k][i][l] if not values else self.Us[k][i][l].X
        ukilp1 = self.Us[k][i][l + 1] if not values else self.Us[k][i][l + 1].X

        delta = ukilp1 - ukil

        slope = delta * self.L / (self.maxs[i] - self.mins[i])
        val = ukil + slope * width
        return val

    def u_k(self, k, X, values: bool = False):
        if not values:
            return quicksum(self.u_k_i(k, i, X, values=False) for i in range(self.n))
        else:
            return sum(self.u_k_i(k, i, X, values=True) for i in range(self.n))

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        self.mins = np.minimum(X.min(axis=0), Y.min(axis=0))
        self.maxs = np.maximum(X.max(axis=0), Y.max(axis=0))

        self.z = []

        # Set min and max values
        for k in range(self.K):
            for i in range(self.n):
                self.model.addConstr(self.Us[k][i][0] == 0)
            self.model.addConstr(
                quicksum(self.Us[k][i][-1] for i in range(self.n)) == 1
            )

        self.or_constraint_variables = []

        self.underestimation_variables = []
        self.overestimation_variables = []

        for index, couple in enumerate(zip(X, Y)):
            self.z.append(self.model.addVar(name=f"one_of_clusters_constraint_{index}"))
            x, y = couple

            var_acc = []
            for k in range(self.K):
                b = self.model.addVar(
                    name=f"binary_indicator_{index}_cluster_{k}", vtype=GRB.BINARY
                )
                splusx = self.model.addVar(name=f"overestimation_x_{index}_cluster_{k}")
                sminusx = self.model.addVar(
                    name=f"underestimation_x_{index}_cluster_{k}"
                )

                splusy = self.model.addVar(name=f"overestimation_y_{index}_cluster_{k}")
                sminusy = self.model.addVar(
                    name=f"underestimation_y_{index}_cluster_{k}"
                )

                self.overestimation_variables.extend((splusx, splusy))
                self.underestimation_variables.extend((sminusx, sminusy))
                self.model.addConstr(splusx >= 0)
                self.model.addConstr(splusy >= 0)
                self.model.addConstr(sminusx >= 0)
                self.model.addConstr(sminusy >= 0)

                M = 100

                self.model.addConstr(
                    (self.u_k(k, x) - splusx + sminusx)
                    - (self.u_k(k, y) - splusy + sminusy)
                    + (1 - b) * M
                    >= 0
                )
                var_acc.append(b)
            self.model.addConstr(self.z[index] == or_(var_acc))
            self.model.addConstr(self.z[index] == 1)
            self.or_constraint_variables.extend(var_acc)

        for k in range(self.K):
            for l in range(self.L - 1):
                for i in range(self.n):
                    self.model.addConstr(
                        self.Us[k][i][l + 1] - self.Us[k][i][l] >= self.epsilon
                    )

        self.model.setObjective(
            quicksum(self.overestimation_variables + self.underestimation_variables),
            GRB.MINIMIZE,
        )
        self.model.optimize()
        return

    def predict_utility(self, X: np.ndarray) -> np.array:
        """Return Decision Function of the MIP for X.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns:
        -----------
        1-D numpy array
        """

        criteria = np.array(
            list(
                map(
                    lambda x: [self.u_k(k, x, values=True) for k in range(self.K)],
                    X,
                )
            )
        )

        return criteria


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_clusters, n_pieces=5, n_criteria=10):
        """Initialization of the Heuristic Model."""
        self.seed = 123
        self.K = n_clusters
        self.L = n_pieces
        self.n = n_criteria
        self.instantiate()
        self.train_samples_limit = 4000

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        self.preprocessing = Pipeline(("std_scaler", StandardScaler()))
        self.k_means = KMeans(n_clusters=self.K)
        self.utility_functions = [k for k in range(self.K)]
        return

    def fit(self, X, Y, labels=None, iterations=40):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        X = X[: self.train_samples_limit]
        Y = Y[: self.train_samples_limit]
        if labels:
            labels = labels[: self.train_samples_limit]
        results = {"explained": [], "grouped": []}
        diff = X - Y

        data_to_be_clustered = diff

        pairs_explained = metrics.PairsExplained()
        cluster_intersection = metrics.ClusterIntersection()

        k_means_init = "k-means++"

        for i in range(iterations):

            print(
                f"################################ ITERATION {i} #################################"
            )
            self.k_means = KMeans(n_clusters=self.K)
            self.k_means.fit(data_to_be_clustered)

            self.UTAs = [
                PairwiseUTA(n_criteria=self.n, n_pieces=self.L) for k in range(self.K)
            ]

            for cluster in range(self.K):
                print(
                    f"------------------ TRAINING CLUSTER {cluster} ------------------ "
                )
                cluster_X = X[lbls == cluster, :]
                cluster_Y = Y[lbls == cluster, :]
                self.UTAs[cluster].fit(cluster_X, cluster_Y)

            row1 = X[0, :].reshape(-1, self.n)

            self.predict_utility(row1)

            eval_X = np.apply_along_axis(self.predict_utility, axis=1, arr=X)
            eval_Y = np.apply_along_axis(self.predict_utility, axis=1, arr=Y)

            if labels is not None:
                explained = pairs_explained.from_model(self, X, Y)
                grouped = cluster_intersection.from_model(self, X, Y, labels)

                results["explained"].append(explained)
                results["grouped"].append(grouped)

            data_to_be_clustered = np.hstack((diff, eval_X - eval_Y))
            if type(k_means_init) == str:
                k_means_init = np.concatenate(
                    (
                        self.k_means.cluster_centers_[0],
                        np.array([0 for i in range(self.K)]),
                    ),
                    axis=None,
                )
            else:
                k_means_init = self.k_means.cluster_centers_[0]

        if labels:
            print(results)

        return

    def fit_eval(self, X, Y, Z, iterations=10):
        self.fit(X, Y, labels=Z, iterations=iterations)

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """

        if len(X.shape) == 1:
            return [self.UTAs[k].predict_utility(X) for k in range(self.K)]

        else:
            criteria = np.apply_along_axis(
                lambda x: [self.UTAs[k].predict_utility(x) for k in range(self.K)],
                axis=1,
                arr=X,
            )

        return criteria
