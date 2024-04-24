import numpy as np

def project_onto_constraints(x, lower_bound, upper_bound):
    """ Project x onto the feasible set defined by the box constraints. """
    return np.clip(x, lower_bound, upper_bound)

def compute_gradient(A, x, b):
    """ Compute the gradient of the objective function. """
    return 2 * A.T @ (A @ x - b)

#TODO Check KKT conditions
def check_convergence(x, projected_x, gradient, tol=1e-5):
    """ Check the convergence criteria based on the KKT conditions. """
    return np.linalg.norm(projected_x - x) < tol and np.linalg.norm(gradient) < tol

def gradient_projection_method_for_QP(A, b, x0, lower_bound, upper_bound, tol=1e-5, max_iter=1000):
    x = x0
    n_iter = 0
    
    while n_iter < max_iter:
        gradient = compute_gradient(A, x, b)
        # Compute the Cauchy point as the projection of x - gradient
        cauchy_point = project_onto_constraints(x - gradient, lower_bound, upper_bound)
        # Check convergence (KKT conditions)
        if check_convergence(x, cauchy_point, gradient, tol):
            break
        # Update x for the next iteration
        x = cauchy_point
        n_iter += 1
    
    return x, n_iter

# Problem setup
n = 10
A = np.random.randn(n, n)  # Change as needed
b = np.random.randn(n)     # Change as needed
lower_bound = np.full(n, -1)  # Replace with actual lower bounds
upper_bound = np.full(n, 1)   # Replace with actual upper bounds
x0 = np.random.randn(n)       # Starting point

# Solve the QP problem
x_star, num_iterations = gradient_projection_method_for_QP(A, b, x0, lower_bound, upper_bound)

print("Solution:", x_star)
print("Number of iterations:", num_iterations)
