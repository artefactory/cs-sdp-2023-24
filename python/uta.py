import math

import numpy as np
from gurobipy import *

# Instanciation du modèle


class PairwiseUTA:
    def __init__(self, n_pieces, n_criteria, precision=0.001, **kwargs):
        """Initialization of the Heuristic Model."""
        self.seed = 123
        self.L = n_pieces
        self.n = n_criteria
        self.epsilon = precision

        self.model = Model("Simple PL modelling")
        self.criteria = [
            [self.model.addVar(name=f"u_{i}_{l}") for l in range(self.L + 1)]
            for i in range(self.n)
        ]

        # maj du modèle
        self.model.update()

    def li(self, i, X):
        x = X[i]
        return math.floor(self.L * (x - self.mins[i]) / (self.maxs[i] - self.mins[i]))

    def xl(self, i, l):
        return self.mins[i] + l * (self.maxs[i] - self.mins[i]) / self.L

    def u_i(self, i, X, evaluate: bool = False):
        get_val = (lambda v: v.X) if evaluate else (lambda v: v)
        x = X[i]
        l = self.li(i, X)

        if x >= self.maxs[i]:
            return get_val(self.criteria[i][-1])

        x_l = self.xl(i, l)
        width = x - x_l

        slope = get_val(self.criteria[i][l + 1]) - get_val(
            self.criteria[i][l]
        ) * self.L / (self.maxs[i] - self.mins[i])
        return get_val(self.criteria[i][l]) + slope * width

    def s(self, X, evaluate: bool = False):
        if not evaluate:
            return quicksum(self.u_i(i, X, evaluate=False) for i in range(self.n))
        else:
            return sum(self.u_i(i, X, evaluate=True) for i in range(self.n))

    def fit(self, X, Y):
        self.mins = np.minimum(X.min(axis=0), Y.min(axis=0))
        self.maxs = np.maximum(X.max(axis=0), Y.max(axis=0))

        n_examples = X.size

        self.sigma_plus = [
            self.model.addVar(name=f"sigma+_{i}") for i in range(n_examples)
        ]
        self.sigma_minus = [
            self.model.addVar(name=f"sigma-_{i}") for i in range(n_examples)
        ]

        # Ajout des contraintes
        # contraintes de préférence des universités
        for i, xy in enumerate(zip(X, Y)):
            x, y = xy
            self.model.addConstr(
                self.s(x) - self.sigma_plus[i] + self.sigma_minus[i]
                >= self.s(y) + self.epsilon
            )

        for sig in self.sigma_minus + self.sigma_plus:
            self.model.addConstr(sig >= 0)

        # contrainte 3
        for l in range(self.L - 1):
            for i in range(self.n):
                self.model.addConstr(
                    self.criteria[i][l + 1] - self.criteria[i][l] >= self.epsilon
                )

        # contrainte 4
        for i in range(self.n):
            self.model.addConstr(self.u_i(i, self.mins) == 0)

        # contrainte 5
        self.model.addConstr(
            quicksum(self.u_i(i, self.maxs) for i in range(self.n)) == 1
        )

        # Fonction Objectif
        self.model.setObjective(
            quicksum(self.sigma_plus) + quicksum(self.sigma_minus), GRB.MINIMIZE
        )
        # Paramétrage (mode mute)
        # Résolution du problème linéaire
        self.model.optimize()

    def predict_utility(self, X):
        return self.s(X, evaluate=True)
