import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import Model, GRB, max_, quicksum

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
        weights_1 = np.random.rand(num_features) # Weights cluster 1
        weights_2 = np.random.rand(num_features) # Weights cluster 2

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
        u_1 = np.dot(X, self.weights[0]) # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1]) # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1) # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, epsilon):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.model = self.instantiate()
        self.L = n_pieces
        self.K = n_clusters
        self.epsilon = epsilon

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        np.random.seed(self.seed)
        model = Model("TwoClustersMIP")
        return model

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        maxs = np.ones(self.n)
        mins = np.zeros(self.n)
        self.n = X.shape[1]
        self.P = X.shape[0]

        #Last index of the piecewise function
        def li(x, i):
            return np.floor(self.L * (x - mins[i]) / (maxs[i] - mins[i]))
        
        #Lower bound of the piecewise function
        def xl(i, l):
            return mins[i] + l * (maxs[i] - mins[i]) / self.L
        
        # Variables of our model
        # Utilitary functions
        self.U = {
            (k, i, l): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="u_{}_{}_{}".format(k, i, l), ub=1)
                for k in range(self.K)
                for i in range(self.n)
                for l in range(self.L+1)
        }
        uik_xij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = li(X[j, i], i)
                    x = xl(i, l)
                    x1 = xl(i, l+1)
                    uik_xij[k, i, j] = self.U[(k, i, l)] + ((X[j, i] - x) / (x1 - x)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uik_yij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = li(Y[j, i], i)
                    x = xl(i, l)
                    x1 = xl(i, l+1)
                    uik_yij[k, i, j] = self.U[(k, i, l)] + ((Y[j, i] - x) / (x1 - x)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uk_xj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_xj[k, j] = quicksum(uik_xij[k, i, j] for i in range(self.n))
        
        uk_yj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_yj[k, j] = quicksum(uik_yij[k, i, j] for i in range(self.n))

        # Binary variables for the preference of X over Y 
        self.delta = {
            (k, j): self.model.addVar(
                vtype=GRB.BINARY, name="delta_{}_{}".format(k, j))
                for k in range(self.K)
                for j in range(self.P)
        }

        # Overestimation  and underestimation of the utilities
        self.sigmaxplus = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmaxplus_{}".format(j), ub=1)
                for j in range(self.P)
        }
        self.sigmaxminus = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmaxminus_{}".format(j), ub=1)
                for j in range(self.P)
        }
        self.sigmayplus = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmayplus_{}".format(j), ub=1)
                for j in range(self.P)
        }
        self.sigmayminus = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigmayminus_{}".format(j), ub=1)
                for j in range(self.P)
        }

        # Constraints
        # The preference of X over Y with the binary variables
        M = 100

        self.model.addConstrs(
            (uk_xj[k, j] - self.sigmaxplus[j] + self.sigmaxminus[j] - uk_yj[k, j] + self.sigmayplus[j] - self.sigmayminus[j] - self.epsilon >= -M*(1-self.delta[(k,j)]) for j in range(self.P) for k in range(self.K))
        )
        # There's at least one cluster that prefers X over Y
        for j in range(self.P):
            self.model.addConstr(
                quicksum(self.delta[(k, j)] for k in range(self.K)) >= 1
            )
        # The function must be ascending
        self.model.addConstrs(
            (self.U[(k, i, l+1)] - self.U[(k, i, l)]>=self.epsilon for k in range(self.K) for i in range(self.n) for l in range(self.L)))

        # Utility starting point
        self.model.addConstrs(
            (self.U[(k, i, 0)] == 0 for k in range(self.K) for i in range(self.n)))
        # Utility function normalization
        self.model.addConstrs(
            (quicksum(self.U[(k, i, self.L)] for i in range(self.n)) == 1 for k in range(self.K)))
        
        # Objective of our model
        self.model.setObjective(quicksum(self.sigmaxplus[j] + self.sigmaxminus[j] + self.sigmayplus[j] + self.sigmayminus[j] for j in range(self.P)), GRB.MINIMIZE)

        #A plot to see the importance of the features
        def plot_Uf_perK(U):
            import matplotlib.pyplot as plt
            for k in range(self.K):
                plt.figure(k)  # Crée une nouvelle figure pour chaque cluster
                for i in range(self.n):
                # Calcule les valeurs de l'axe des x et de l'axe des y pour la caractéristique i et le cluster k
                    x_values = [xl(i, l) for l in range(self.L+1)]
                    y_values = [U[k, i, l] for l in range(self.L+1)]
                    plt.plot(x_values, y_values, label="feature {}".format(i))
            plt.title(f"Utility functions for cluster {k}")
            plt.legend()
            plt.xlabel("Feature Value")
            plt.ylabel("Utility")
            plt.grid(True)  # Ajoute une grille pour faciliter la lecture
            plt.show()  # Affiche la figure

        # Solving our problem
        self.model.params.outputflag = 0  # Mute mode for Gurobi
        self.model.update()
        self.model.optimize()
        if self.model.status == GRB.INFEASIBLE:
            print("\n Unfeasible")
            raise Exception("Unfeasible")
        elif self.model.status == GRB.UNBOUNDED:
            print("\n Unbounded")
            raise Exception("Unbounded")
        else:
            print("\n Solution found")
            # print the value of objective function
            print("objective function value: ", self.model.objVal)
            self.U = {(k, i, l): self.U[k, i, l].x for k in range(self.K) for i in range(self.n) for l in range(self.L+1)}
            self.delta = {(k, j): self.delta[k, j].x for k in range(self.K) for j in range(self.P)}

            sampleX = np.expand_dims(X[0], axis=0)
            sampleY = np.expand_dims(Y[0], axis=0)
            utilityX = self.predict_utility(sampleX)
            utilityY = self.predict_utility(sampleY)
            print("utility X: ", utilityX)
            print("utility Y: ", utilityY) 
            plot_Uf_perK(self.U)
        return self

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
        maxs = np.ones(self.n)
        mins = np.zeros(self.n)
        A = X.shape[0]

        def li(x, i):
            return int(np.floor(self.L * (x - mins[i]) / (maxs[i] - mins[i])))

        
        def xl(i, l):
            return mins[i] + l * (maxs[i] - mins[i]) / self.L
        
        utilities = np.zeros((A, self.K))
        for k in range(self.K):
            for j in range(A):
                for i in range(self.n):
                    l = li(X[j, i], i)
                    utilities[j, k] += self.U[k, i, li(X[j, i], i)] + ((X[j, i] - xl(i, li(X[j, i], i))) / (xl(i, li(X[j, i], i)+1) - xl(i, li(X[j, i], i)))) * (self.U[k, i, li(X[j, i], i)+1] - self.U[k, i, li(X[j, i], i)])

        return utilities

class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
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
