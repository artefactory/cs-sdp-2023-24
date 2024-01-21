from gurobipy import *
import math
import numpy as np

X = np.load("cs-sdp-2023-24/data/dataset_4/X.npy")
Y = np.load("cs-sdp-2023-24/data/dataset_4/Y.npy")

# Instanciation du modèle
m = Model("First model")

e = 0.001
e2 = 1e-6
K = 2
n = 4
L = 5
P = 400
N = 2000
M1 = 1.1
M2 = 2.1

U = [[[m.addVar(name=f"u_{k}_{i}_{l}") for l in range(1,L)] for i in range(n)] for k in range(K)]
alpha = [[m.addVar(vtype=GRB.BINARY, name=f"a_{j}_{k}") for k in range(K)] for j in range(N)]
beta = [m.addVar(vtype=GRB.BINARY, name=f"b_{j}") for j in range(N)]

m.update()

minsX = X.min(axis=0)
maxsX = X.max(axis=0)

print(minsX)


minsY = Y.min(axis=0)
maxsY = Y.max(axis=0)

mins = [min(minsX[i], minsY[i]) for i in range(n)]
maxs = [max(maxsX[i], maxsY[i]) for i in range(n)]

U_abscisse = [[mins[i] + l * (maxs[i] - mins[i]) / L for l in range(L+1)] for i in range(n)]

#functions
def lineaire_morceaux(X,Y,x0):
    """Renvoie l'ordonnée y0 d'un point d'abscisse x0 sur la courbe
    passant par les points de coordonnées (X[i],Y[i])"""
    i = 0
    while X[i] - x0 <= -e2:
        i += 1
    i -= 1
    return Y[i-1] + (Y[i]-Y[i-1])/(X[i]-X[i-1])*(x0-X[i-1])

def U_k_i(k,i,x_j):
    """Renvoie la valeur de la fonction U_k_i au point d'abscisse x"""
    x = x_j[i]
    Y = [0] + [U[k][i][l] for l in range(L-1)]
    X = U_abscisse[i]
    return lineaire_morceaux(X,Y,x)

def U_k(k,x_j):
    """Renvoie la valeur de la fonction U_k au point d'abscisse x"""
    return sum([U_k_i(k,i,x_j) for i in range(n)])

#add constraints
for j in range(N):
    for k in range(K):
        m.addConstr(U_k(k,X[j]) - U_k(k,Y[j]) - M1*alpha[j][k] <= -e)
        m.addConstr(U_k(k,X[j]) - U_k(k,Y[j]) - M1*(alpha[j][k] - 1) >= 0)

    m.addConstr(sum([alpha[j][k] for k in range(K)]) - 1 - M2*beta[j] <= -e)
    m.addConstr(sum([alpha[j][k] for k in range(K)]) - 1 - M2*(beta[j] - 1) >= 0)

for k in range(K):
    for i in range(n):
        m.addConstr(U[k][i][0] >= 0)
        for l in range(L-2):
            m.addConstr(U[k][i][l] - U[k][i][l+1] <= 0)

    m.addConstr(sum([U[k][i][L-2] for i in range(n)]) == 1)
  

#set objective
m.setObjective(sum([beta[j] for j in range(N)]), GRB.MAXIMIZE)
m.optimize()


#save solution
U_sol = [[[U[k][i][l].x for l in range(L-1)] for i in range(n)] for k in range(K)]
alpha_sol = [[alpha[j][k].x for k in range(K)] for j in range(N)]
beta_sol = [beta[j].x for j in range(N)]

np.save("cs-sdp-2023-24/data/dataset_4/U_sol.npy", U_sol)
np.save("cs-sdp-2023-24/data/dataset_4/alpha_sol.npy", alpha_sol)
np.save("cs-sdp-2023-24/data/dataset_4/beta_sol.npy", beta_sol)
