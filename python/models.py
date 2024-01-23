import pickle
from gurobipy import *
from abc import abstractmethod

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
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.model = self.instantiate(n_clusters, n_pieces)

    def instantiate(self,n_clusters, n_pieces):
        """Instantiation of the MIP Variables - To be completed."""
        # Instanciation du modèle
        m = Model("First model")

        # Constants
        self.e = 0.001
        self.e2 = 1e-6
        self.K = 2
        self.L = 5
        self.P = 400
        self.M1 = 1.1
        self.M2 = 2.1

        return m


    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        self.N = len(X)
        self.n = X.shape[1]

        self.U = [[[self.model.addVar(name=f"u_{k}_{i}_{l}") for l in range(1,self.L)] for i in range(self.n)] for k in range(self.K)]
        self.alpha = [[self.model.addVar(vtype=GRB.BINARY, name=f"a_{j}_{k}") for k in range(self.K)] for j in range(self.N)]
        # self.beta = [self.model.addVar(vtype=GRB.BINARY, name=f"b_{j}") for j in range(self.N)]
        self.sigma_pos = [[self.model.addVar( name=f"sigma_pos_{j}_{k}") for k in range(self.K)] for j in range(self.N)]
        self.sigma_neg = [[self.model.addVar( name=f"sigma_neg_{j}_{k}") for k in range(self.K)] for j in range(self.N)]

        self.model.update()

        minsX = X.min(axis=0)
        maxsX = X.max(axis=0)
        minsY = Y.min(axis=0)
        maxsY = Y.max(axis=0)
        maxs = [max(maxsX[i], maxsY[i]) for i in range(self.n)]
        mins = [min(minsX[i], minsY[i]) for i in range(self.n)]

        self.U_abscisse = [[mins[i] + l * (maxs[i] - mins[i]) / self.L for l in range(self.L+1)] for i in range(self.n)]

        #add constraints
        for j in range(self.N):
            for k in range(self.K):
                self.model.addConstr(self.U_k(k,X[j],self.U) - self.U_k(k,Y[j],self.U) + self.sigma_pos[j][k] - self.sigma_neg[j][k] - self.M1*self.alpha[j][k] <= -self.e)
                self.model.addConstr(self.U_k(k,X[j],self.U) - self.U_k(k,Y[j],self.U) + self.sigma_pos[j][k] - self.sigma_neg[j][k] - self.M1*(self.alpha[j][k] - 1) >= 0)
                self.model.addConstr(self.sigma_pos[j][k] >= 0)
                self.model.addConstr(self.sigma_neg[j][k] >= 0)
            self.model.addConstr(sum([self.alpha[j][k] for k in range(self.K)]) >= 1)

            # self.model.addConstr(sum([self.alpha[j][k] for k in range(self.K)]) - 1 - self.M2*self.beta[j] <= -self.e)
            # self.model.addConstr(sum([self.alpha[j][k] for k in range(self.K)]) - 1 - self.M2*(self.beta[j] - 1) >= 0)


        for k in range(self.K):
            for i in range(self.n):
                self.model.addConstr(self.U[k][i][0] >= 0)
                for l in range(self.L-2):
                    self.model.addConstr(self.U[k][i][l] - self.U[k][i][l+1] <= 0)

            self.model.addConstr(sum([self.U[k][i][self.L-2] for i in range(self.n)]) == 1)
        
        

        #set objective
        self.model.setObjective(sum([sum([self.sigma_neg[j][k] + self.sigma_pos[j][k] for k in range(self.K)]) for j in range(self.N)]), GRB.MINIMIZE)
        self.model.optimize()
        
        print("Fit done !" + str(self.model.status) + " " + str(self.model.ObjVal))

        self.U_sol = [[[self.U[k][i][l].x for l in range(self.L-1)] for i in range(self.n)] for k in range(self.K)]
        # self.alpha_sol = [[self.alpha[j][k].x for k in range(self.K)] for j in range(self.N)]
        # self.beta_sol = [self.beta[j].x for j in range(self.N)]

    
    def lineaire_morceaux(self, X,Y,x0):
        """Renvoie l'ordonnée y0 d'un point d'abscisse x0 sur la courbe
        passant par les points de coordonnées (X[i],Y[i])"""
        i = 0
        while X[i] - x0 <= -self.e2:
            i += 1
            if i == self.L:
                return Y[i-1]
        i -= 1
        return Y[i-1] + (Y[i]-Y[i-1])/(X[i]-X[i-1])*(x0-X[i-1])

    def U_k_i(self, k,i,x_j,U_coef):
        """Renvoie la valeur de la fonction U_k_i au point d'abscisse x"""
        x = x_j[i]
        Y = [0] + [U_coef[k][i][l] for l in range(self.L-1)]
        X = self.U_abscisse[i]
        return self.lineaire_morceaux(X,Y,x)

    def U_k(self, k,x_j,U_coef):
        """Renvoie la valeur de la fonction U_k au point d'abscisse x"""
        return sum([self.U_k_i(k,i,x_j,U_coef) for i in range(self.n)])

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        result = []
        for x in X:
            result.append([self.U_k(0,x,self.U_sol), self.U_k(1,x,self.U_sol)])
        return np.array(result)


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
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return
