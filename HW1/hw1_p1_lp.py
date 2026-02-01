import gurobipy as gp
from gurobipy import GRB
import numpy as np
import itertools as it


def build_model(A, b, c):
    """
    Input:
        A: (m × n) NumPy array
        b: length-m NumPy array
        c: length-n NumPy array

    Output:
        A Gurobi model for the LP:

            maximize     c^T x
            subject to   A x <= b
                          x >= 0

        where x has n nonnegative decision variables.
    """
    _, n = A.shape 
    model= gp.Model()
    x = model.addMVar(n, lb=0)
    model.setObjective(c @ x, GRB.MAXIMIZE)
    model.addConstr(A @ x <= b)
    return model 

def solve_with_gurobi(A, b, c):
    """
    Input:
        A: (m × n) NumPy array
        b: length-m NumPy array
        c: length-n NumPy array

    Builds the model using build_model(A, b, c),
    solves it with Gurobi, and returns (x_opt, obj_val):
        x_opt: optimal solution vector (NumPy array length n)
        obj_val: optimal objective value (float)
    """
    _, n = A.shape
    model = build_model(A, b, c)
    model.optimize() 
    return np.hstack([(model.getVars()[i]).X for i in range(n)]), model.ObjVal

  
def enumerate_vertices(A, b, c):
    """
    Enumerates all vertices (extreme points) of the 2D polytope

          P = { x in R^2 : A x <= b, x>=0 },

    and returns:

        vertices: list of NumPy arrays (each of length 2)
        best_x:   vertex achieving max c^T x subject to A x <= b and x>=0 from vertices
        best_obj: the corresponding objective value of best_x

    Assumptions:
        - A is an (m × 2) matrix.
        - Vertices are finite.

    This must work for *arbitrary* 2D LPs, not just the example.
    """  
    m, n = A.shape
    eqIdx = np.arange(m)
    pairs = it.combinations(eqIdx, 2)
    vertices = []
    for pair in pairs:
        idx1 = pair[0].item()
        idx2 = pair[1].item()
        x = np.linalg.solve(A[[idx1,idx2]], b[[idx1, idx2]])
        if ((x >= 0).all() and (A @ x >= 0).all()): #check feasiblity
            vertices.append(x)
    vertices = np.hstack(vertices)
    best_x, best_obj = solve_with_gurobi(A, b, c)
    return vertices, best_x, best_obj
        
if __name__ == "__main__":
    c = np.array([3,2])
    A = np.array([[1,1],[2,1]])
    b = np.array([4,5])
    vertices, opt_x, _ = enumerate_vertices(A, b, c)
    print(vertices, opt_x)
   


