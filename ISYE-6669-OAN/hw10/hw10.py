import cvxpy as cp
import numpy as np


def solve_rmp(A, n):
    """[Solve reduced master problem]

    Args:
        A ([type]): [A matrix containing weights for each roll type]
        n ([type]): [Number of decision variables for x]
        c ([type]): [Cost vector with reduced costs for each iteration]
        b ([type]): [Bounds]
    """    
    x = cp.Variable((n,1))
    
    b = np.array([15, 30, 20]).reshape((3,1))
    c = np.ones((3,1))
    # defining objective
    objective = cp.Minimize(cp.sum(x))

    # defining constraints
    constraints = [A@x==b,
              x>=0]

    rmp = cp.Problem(objective, constraints)
    rmp.solve()
    # printing outputs
    print("\nThe optimal value for the RMP objective function is:", round(rmp.value, 2))
    print("\nThe values for variable x's in RMP are:", rmp.variables()[0].value)

    return rmp


def calculate_reduced_costs(B):
    """[Calculate reduced costs for direction vector]

    Args:
        y ([np.array]): [Cost vector]
        B ([np.array]): [Basis matrix]

    Returns:
        [type]: [Direction db vector with weights to determine if we're reached our optimal solution, 
        if all weights > 0, we've reached optimal, if any weights < 0 we select the minimum index to enter
        enter the basis]
    """    
    #Cb 
    c = np.ones((3,1))
    #Get inverse of basis matrix
    B_inv = np.linalg.inv(B)
    #Get reduced costs for simplex method step
    y = B_inv.dot(c)
    #Return index of minimum value
    print("\nThe reduced costs are:", y)
    return y

def solve_knapsack_problem(y):
    #Knapsack problem
    a = cp.Variable((3,1), integer=True)
    w = np.array([7, 11, 16]).reshape((3,1))
    
    knapsack_objective = cp.Maximize(cp.sum(cp.multiply(a,y)))
    # defining constraints
    knap_sack_constraints = [cp.sum(cp.multiply(w,a))<=80,
                a>=0]
    knapsack_prob = cp.Problem(knapsack_objective, knap_sack_constraints)
    knapsack_prob.solve()

    # printing outputs
    print("\nThe optimal value for knapsack objective function is:", round(knapsack_prob.value, 2))
    print("\nThe values for variable a's in knapsack are:", knapsack_prob.variables()[0].value)
    return knapsack_prob


if __name__ == '__main__':
    #RM_Step 1
    A_1 = np.array([[10, 0, 0],              
                [0, 7, 0],              
                [0, 0, 5]])
    A_2 = np.array([[11,10, 0, 0],
                    [0,0, 7, 0],              
                    [0,0, 0, 5]])

    # defining objective
    rmp = solve_rmp(A_1,3)

    #Determine which variables are non-negative from the optimal solution, to create basis matrix
    basis = A_1[:,np.where(rmp.variables()[0].value >= 1e-5)[0]]
    y_reduced_costs = calculate_reduced_costs(basis)

    #Calculate knapsack problem
    knapsack_sol = solve_knapsack_problem(y_reduced_costs)

    # defining objective
    rmp_step2 = solve_rmp(A_2,4)

    #Determine which variables are non-negative from the optimal solution, to create basis matrix
    basis_2 = A_2[:,np.where(rmp_step2.variables()[0].value >= 1e-5)[0]]
    y_reduced_costs_2 = calculate_reduced_costs(basis_2)

    #Calculate knapsack problem
    knapsack_sol2 = solve_knapsack_problem(y_reduced_costs_2)
