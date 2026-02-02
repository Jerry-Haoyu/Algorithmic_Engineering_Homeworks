
# from hw1_p1_lp import build_model, solve_with_gurobi
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
# from io import StringIO

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

def solve_diet(df, requirements):
    """
    Solve the diet LP:
    minimize total cost (USD)
    subject to daily nutritional requirements
    Inputs
    ------
    df : pandas.DataFrame
    Each row is a food. Must contain the columns:
    - "food_one_serving"
    - "price_usd_per_serving"
    - "calories_kcal"
    - "protein_g"
    - "fiber_g"
    - "sugar_g"
    - "fat_g"
    - "sodium_mg"
    requirements : dict
    With keys:
    - "calories_min"
    - "protein_min"
    - "fiber_min"
    - "sugar_max"
    - "fat_max"
    - "sodium_max"
    Outputs
    -------
    servings : np.ndarray, shape (len(df),)
    servings[i] is the number of servings of row i in df.
    min_cost : float
    The minimum total cost (objective value).
    """
    calories_min = requirements["calories_min"]
    protein_min = requirements["protein_min"]
    fiber_min = requirements["fiber_min"]
    sugar_max = requirements["sugar_max"]
    fat_max = requirements["fat_max"]
    sodium_max = requirements["sodium_max"]
    
    n = len(df) # number of foods
    prices = df["price_usd_per_serving"].values # NumPy array of length n
    c = -prices #to maximize -prices is to minimize prices
    
    calories = df["calories_kcal"].values
    protein = df["protein_g"].values
    fiber = df["fiber_g"].values
    sugar = df["sugar_g"].values
    fat = df["fat_g"].values
    sodium = df["sodium_mg"].values
    A = np.vstack([-calories, -protein, -fiber, sugar, fat, sodium])
    b = np.array([-calories_min, -protein_min, -fiber_min, sugar_max, fat_max, sodium_max])
    x_opt, obj_opt = solve_with_gurobi(A, b, c)
    return x_opt, -obj_opt 

if __name__ == "__main__":
    csv = StringIO("""food_one_serving,price_usd_per_serving,calories_kcal,protein_g,carbs_g,sugar_g,fiber_g,fat_g,sodium_mg
                chicken,1.80,128,24,0,0,0,2.7,44
                banana,0.30,105,1.3,27,14,3.1,0.4,1
                yogurt,0.90,104,5.9,7.9,7.9,0,5.5,70
                beans,1.10,120,8,21,1,7,0.5,2
                spinach,0.40,7,0.9,1.1,0.1,0.7,0.1,24
                almonds,0.70,160,6,6,1,3,14,1
            """)
    df = pd.read_csv(csv)
    requirements = {
        "calories_min": 2000,
        "protein_min": 100,
        "fiber_min": 50,
        "sugar_max": 50,
        "fat_max": 120,
        "sodium_max": 230
    }
    print(solve_diet(df, requirements))