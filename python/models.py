import pickle
from abc import abstractmethod
from gurobipy import Model, GRB
import gurobipy as gp

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

    def __init__(self, n_pieces, n_clusters):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.n_pieces = n_pieces
        self.n_clusters = n_clusters
        self.model = self.instantiate()
        self.w = None
        self.g = np.linspace(0, 1, self.n_pieces + 1)
        

    def instantiate(self):
        """Instantiation of the MIP Variables """
        # Create new model
        m = Model("TwoClustersMIP")

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

        # Define parameters
        
        n_pairs, n_features = X.shape
        # Breaking points g
        g = self.g

        # Define alpha and the breaking points
        alpha_x = {}
        alpha_y = {}
        low_up_idx_x = {}
        low_up_idx_y = {}

        for p in range(n_pairs):
            for i in range(n_features):
                # Find the index of the breaking points
                up_idx_x = np.searchsorted(g, X[p, i])
                up_idx_y = np.searchsorted(g, Y[p, i])
                low_idx_x = up_idx_x - 1
                low_idx_y = up_idx_y - 1
                
                # Store the index of the breaking points
                low_up_idx_x[p, i] = (low_idx_x, up_idx_x)
                low_up_idx_y[p, i] = (low_idx_y, up_idx_y)
                
                # Compute alpha
                alpha_x[p, i] = (X[p,i] - g[low_idx_x]) / (g[up_idx_x] - g[low_idx_x])
                alpha_y[p, i] = (Y[p,i] - g[low_idx_y]) / (g[up_idx_y] - g[low_idx_y])

        # Big M
        M = 1000
        # Little delta
        delta = 0.01
        delta_2 = 0.001

        # Define variables
        # Assuming number of clusters (k), number of pieces (l), 
        # number of pairs (p), and number of criterio (i) that come from your problem definition.

        # Create the w^{k,l}_i: weights of the utility function of each feature i for each cluster k and each piece l (breakingpoint)
        w = {}
        for k in range(self.n_clusters):
            for l in range(self.n_pieces + 1):
                for i in range(n_features):
                    w[k, l, i] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w_{k}_{l}_{i}")

        # Create the σ_pk: marginal error of each pair p for each cluster k
        sigma = {}
        for p in range(n_pairs):
            for k in range(self.n_clusters):
                sigma[p, k] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"sigma_{p}_{k}")

        # Create the v_pk: binary variable indicating if pair p is explained by cluster k
        v = {}
        for p in range(n_pairs):
            for k in range(self.n_clusters):
                v[p, k] = self.model.addVar(vtype=GRB.BINARY, name=f"v_{p}_{k}")

        # Integrate new variables
        self.model.update()

        # Constrains

        # Each pair p is valid for at least one cluster k
        for p in range(n_pairs):
            self.model.addConstr(gp.quicksum(v[p, k] for k in range(self.n_clusters)) >= 1, name=f"valid_pair_{p}")

        # It is valid when the difference of utility between the two elements of the pair is greater than the marginal error
        for p in range(n_pairs):
            for k in range(self.n_clusters):
                
                Ux = gp.quicksum(w[k, low_up_idx_x[p, i][0], i] * (1 - alpha_x[p, i]) + w[k, low_up_idx_x[p, i][1], i] * alpha_x[p, i] for i in range(n_features))
                Uy = gp.quicksum(w[k, low_up_idx_y[p, i][0], i] * (1 - alpha_y[p, i]) + w[k, low_up_idx_y[p, i][1], i] * alpha_y[p, i] for i in range(n_features))

                # Create constraint 1
                self.model.addConstr(Ux - Uy + sigma[p, k] + delta_2 <= v[p, k] * M, name=f"UTA_1_{p}_{k}")

                # Create constraint 2
                self.model.addConstr(Ux - Uy + sigma[p, k] >=  delta - (1 - v[p, k]) * M, name=f"UTA_2_{p}_{k}")
        
        # Monothonicity of preferences 
        for i in range(n_features):
            for k in range(self.n_clusters):
                for l in range(self.n_pieces):
                    self.model.addConstr(w[k, l, i] <= w[k, l+1, i], name=f"mono_{i}_{k}_{l}")
        
        # Normalization of weights
        for k in range(self.n_clusters):
            self.model.addConstr(gp.quicksum(w[k, self.n_pieces, i]  for i in range(n_features)) == 1, name=f"norm_{k}")
        
        # First weight is 0
        for k in range(self.n_clusters):
            self.model.addConstr(gp.quicksum(w[k, 0, i] for i in range(n_features))== 0, name=f"first_{k}_{i}")
        
        # Set objective
        self.model.setObjective(gp.quicksum(sigma[p, k] for p in range(n_pairs) for k in range(self.n_clusters)), GRB.MINIMIZE)

        # Optimize model
        self.model.optimize()

        # Print 10 first weights
        for k in range(self.n_clusters):
            for l in range(self.n_pieces + 1):
                for i in range(n_features):
                    print(f"w_{k}_{l}_{i} = {w[k, l, i].x}")

        # Save variables
        self.w = w 
        self.g = g
        
        # To be completed
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
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.

        u_1 = np.dot(X, self.weights[0]) # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1]) # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1) # Stacking utilities over cluster on axis 1


    def save_model(self, path):
        # Save gurobi model
        self.model.write(path)
    
    def load_model(self, path):
        # Load gurobi model
        self.model = gp.read(path)
        # Save variable w
        self.w = {}
        for v in self.model.getVars():
            if v.varName[0] == 'w':
                self.w[v.varName] = v.x
                
        return self.model


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
    
    
