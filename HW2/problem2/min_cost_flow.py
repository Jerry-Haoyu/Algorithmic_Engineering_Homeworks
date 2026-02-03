
# from hw1_p1_lp import build_model, solve_with_gurobi
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
# from io import StringIO

def solve_gridlock(nodes, arcs, supply):
    """
    Solve a minimum-cost flow LP.
    Inputs
    ------
    nodes : list[str]
    Node names.
    arcs : pd.DataFrame
    Columns: 'from', 'to', 'capacity', 'cost'.
    supply : dict[str, float]
    Node supplies b_n (positive) and demands (negative).
    Must satisfy sum(b_n) = 0.
    Returns
    -------
    flow : dict[(str, str), float]
    flow[(u,v)] is the optimal flow on arc (u,v).
    total_cost : float
    Minimum total transportation cost.
    9
    """
    m, n = arcs.shape #number of nodes, number of edges 
    model = gp.Model("gridlock") 
    
    #flow, note there are |edges| number of flow, obviously 
    f =  model.addMVar(m, lb=0, name="flow") 
    
    start_nodes : np.ndarray = arcs["from"].values 
    end_nodes : np.ndarray = arcs["to"].values 
    capacities : np.ndarray = arcs["capacity"].values
    costs : np.ndarray = arcs["cost"].values
    
    #objective to minimize cost 
    model.setObjective(f @ costs, GRB.MINIMIZE)
    
    #capacity conservation, LB of 0 is enforced by addMVar(m, lb=0)
    model.addConstr(f <= capacities, "capacity conservation")
    
    #flow conservation constraints
    in_map : dict[str, list[int]] = {}  #map from node to edge index
    out_map : dict[str, list[int]] = {}
    
    for node in nodes :
        in_map[node] = []
        out_map[node] = []
        
    for j in range(m):
        u = start_nodes[j]
        v = end_nodes[j]
        in_map[v].append(j)
        out_map[u].append(j)
    
    for node in nodes:
        in_nodes_flow = f[in_map[node]]
        out_nodes_flow = f[out_map[node]]
        model.addConstr((-in_nodes_flow.sum() + out_nodes_flow.sum()) == supply[node], "flow conservation")
    
    model.optimize()   
    
    flow : dict[(str, str), float] = {} 
    for j in range(m):  
        flow[(start_nodes[j], end_nodes[j])] = f[j].X
    
    return flow, model.ObjVal

if __name__ == "__main__":
    nodes = ["A", "B", "C", "D"]
    supply = {"A" : 3, "B" : 2, "C" : 1, "D" : -6}
    arcs = pd.read_csv("/Users/jerrytang/Documents/CS/CS498_Algorithmic_Engineering/Algorithmic_Engineering_HWs/HW2/problem2/test1.csv")
    flow, opt = solve_gridlock(nodes, arcs, supply)
    print("flow is", flow)
    print("opt is", opt)