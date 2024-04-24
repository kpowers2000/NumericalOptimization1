import numpy as np
import matplotlib.pyplot as plt
from util_functions.backtracking import backtracking as backtracking
#I want to minimize the function f(x) = ||  Ax - b ||^2 = sum_i (a_i^T x - b_i)^2
#where A = [[a1,a2], [a3,a4]] and b = [b1,b2]

global n
global A
global b
global upper_bound
global lower_bound
### HELPER FUNCTIONS ###
def f_obj(x):
    global n
    global A
    global b
    # A is nxn and b is nx1 so x is nx1
    #f = ||  Ax - b ||^2 = sum_i (a_i^T x - b_i)^2
    f = 0
    for i in range(n):
        f += (np.dot(A[i],x) - b[i])**2
        #pause debugger here
        #import pdb; pdb.set_trace()
    
    return f
def grad_f(x):
    global A
    global b
    #grad(f(x)) = 2A^TAx-2A^Tb
    return 2 * A.T @ (A @ x - b)

### Reusing compute_grad_and_obj ###
def compute_grad_and_obj(x):
    f = f_obj(x)
    gradient = grad_f(x)
    return f, gradient

def gradient_projection_method(n_val,A_val,b_val,upper_bound_val,lower_bound_val,x,answer):
    global n
    global A
    global b
    global upper_bound
    global lower_bound
    n = n_val
    A = A_val
    b = b_val
    lower_bound = lower_bound_val
    upper_bound = upper_bound_val
    tol = 1e-3
    max_iter = 200000
    error = 1
    iter = 0
    rho = 0.8
    c = 1e-4

    error_hist = []
    f_values = []
    x_values = []
    alpha_k_values = []
    gradient_hist = []

    f = f_obj(x)
    gradient = grad_f(x)
    #Store initial values
    f_0 = f
    x_0 = x
    grad_f_0_norm = np.linalg.norm(gradient)
    #let e_k = ||x_k - x*||, where x* is the minimizer
    #x* = [1,1]
    e_k = []
    while error > tol and iter < max_iter:
        iter += 1
        p=0
        p = -gradient
        e_k.append(np.linalg.norm(x - answer))
        alpha_k = alpha_k = backtracking(compute_grad_and_obj, p, x, rho, c)
        s = alpha_k * p
        x_old = x
        x = x+ s
        x = np.clip(x, lower_bound, upper_bound)
        step = abs(x-x_old)
        error = np.linalg.norm(step)/np.linalg.norm(x)
        f, g_new = compute_grad_and_obj(x)
        gradient = g_new

        # Storing values for plotting
        error_hist.append(error)
        f_values.append(f)
        x_values.append(x.copy())
        alpha_k_values.append(alpha_k)
        gradient_hist.append(np.linalg.norm(gradient))
    f_final = f_obj(x)
    #final error
    fin_error = np.linalg.norm(x - answer)
    return x, iter,f_final, fin_error 
import numpy as np

def setup_example_1():
    A = np.array([[1, 0], [0, 1]])
    b = np.array([3, 3])
    lower_bound = np.array([-1, -1])
    upper_bound = np.array([2, 2])
    return A, b, lower_bound, upper_bound

def setup_example_2():
    A = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])
    b = np.array([8, 15, 20])
    lower_bound = np.array([1, 0, 3])
    upper_bound = np.array([4, 5, 5])
    return A, b, lower_bound, upper_bound
def paper_example_1(): 
    n = 10
    A = np.eye(n)
    b = np.arange(11, 21)  # Creates an array from 11 to 20
    lower_bound = np.arange(5, 15)
    upper_bound = np.full(n, 15)
    x_initial = np.full(n, 10)
    return A, b, lower_bound, upper_bound, x_initial, n
def paper_example_2(n):
    #seed random
    #np.random.seed(3)
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    A = (A + A.T) / 2  # To make A symmetric
    #make A positive definite`
    A = make_positive_definite(A)

    lower_bound = np.random.randint(-10, 0, size=n)
    lower_bound = np.full(n, -5)
    upper_bound = np.random.randint(10, 20, size=n)
    upper_bound = np.full(n, 15)
    #x_initial is in the middle of the bounds
    x_initial = (lower_bound + upper_bound) / 2
    return A, b, lower_bound, upper_bound, x_initial,n

def diagonal_preconditioner(A):
    # A is your covariance matrix
    D = np.diag(np.diag(A))
    D_inv = np.linalg.inv(D)
    return D_inv
def make_positive_definite(A):
    eigenvalues = np.linalg.eigvalsh(A)
    min_eigenvalue = eigenvalues.min()
    shift = -min_eigenvalue + 1e-6  # Add a small buffer to ensure strict positivity
    A_pos_def = A + np.eye(A.shape[0]) * shift
    return A_pos_def

import scipy.linalg
def ilu_preconditioner(A):
    A_sparse = scipy.sparse.csr_matrix(A)
    ilu = scipy.sparse.linalg.spilu(A_sparse)
    M = scipy.sparse.linalg.LinearOperator(A.shape, ilu.solve)
    return M
def spectral_sqrt_preconditioner(A):
    eigvals, eigvecs = np.linalg.eigh(A)
    Lambda_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    M = eigvecs @ Lambda_inv_sqrt @ eigvecs.T
    return M
def moderated_spectral_preconditioner(A, scale=0.4):
    eigvals, eigvecs = np.linalg.eigh(A)
    # Adjust eigenvalues less aggressively
    moderated_lambda = 1.0 / np.sqrt(eigvals + scale)
    M = eigvecs @ np.diag(moderated_lambda) @ eigvecs.T
    return M
def diagonally_adjusted_initial_guess(x_initial, D):
    # Assuming D is the diagonal matrix used for preconditioning
    return D @ x_initial
def paper_example_3(n):
    #seed random
    np.random.seed(0)
    A = np.random.randn(n, n)
    b = np.random.randn(n)
    A = (A + A.T) / 2  # To make A symmetric
    A += np.eye(n) * np.trace(A) # To make A pos def
    #add preconditioner
    M_inv = diagonal_preconditioner(A)
    A_og = A
    A = M_inv @ A @ M_inv

    lower_bound = np.random.randint(-10, 0, size=n)
    upper_bound = np.random.randint(10, 20, size=n)
    x_initial = np.random.randint(lower_bound, upper_bound, size=n)  # Ensure initial is feasible
    return A, b, lower_bound, upper_bound, x_initial,n, A_og


    # Define lower and upper bounds for each feature
    lower_bound = np.full((n_features, 1), -1)  # Lower bounds for coefficients
    upper_bound = np.full((n_features, 1), 1)   # Upper bounds for coefficients
    
    # Generate an initial feasible solution within the bounds
    x_initial = np.zeros((n_features, 1))  # Start with zero initial guess
    return A, b, lower_bound, upper_bound, x_initial, n_samples
'''
# Example usage
A1, b1, l1, u1 = setup_example_1()
A2, b2, l2, u2 = setup_example_2()
print('Example 1')
gradient_projection_method(2,A1,b1,u1,l1,np.array([0.001,0.001]),np.array([2,2]))
print('Example 2')
gradient_projection_method(3,A2,b2,u2,l2,np.array([1.5,2.5,4]),np.array([4,5,5]))
A3, b3, l3, u3, x3, n3 = paper_example_1()
print('Paper Example 1')
gradient_projection_method(n3,A3,b3,u3,l3,x3,np.arange(11, 21))
n = 100
A4, b4, l4, u4, x4, n4 = paper_example_2(n)
print('Paper Example 2')
x_ans, iter_val,f_base_ans, throwaway  = gradient_projection_method(n4,A4,b4,u4,l4,x4,np.zeros(n))

#Example 2 with Diagonal Preconditioner
# print('Paper Example 2 with Diagonal Preconditioner')
# M_inv = diagonal_preconditioner(A4)
# A5 = M_inv @ A4 @ M_inv
# cond_A = np.linalg.cond(A4)
# diag_cond_precond_A = np.linalg.cond(A5)
# print('Condition number of A: ', cond_A)
# print('Condition number of diagonally preconditioned A: ', diag_cond_precond_A)
# x5 = diagonally_adjusted_initial_guess(x4, M_inv)
# #x, num_iter, f_values = gradient_projection_method(n4,A5,b4,u4,l4,x5,np.arange(11, 21))
# # #Decondition A5 by doing A5 = M @ A5 @ M where M is inverse of M_inv
# # A = A4
# # f_val = f_obj(x)
# # print('actual f: ', f_val)



#Same as above but with moderated spectral preconditioner
print('Paper Example 2 with Moderated Spectral Preconditioner')
S_moderated = moderated_spectral_preconditioner(A4)
A7 = S_moderated @ A4 @ S_moderated
moderated_spectral_cond_precond_A = np.linalg.cond(A7)
cond_A = np.linalg.cond(A4)
print('Condition number of A: ', cond_A)
print('Condition number of moderated spectral preconditioned A: ', moderated_spectral_cond_precond_A)
x_precond, num_iter, f_value, x_error = gradient_projection_method(n4,A7,b4,u4,l4,x4,x_ans)
A = A4
print('x error: ', x_error)
f_error = f_value - f_base_ans
print('f error: ', f_error)


#run it paper example 2 10 times with 10 different random seeds
#store the number of iterations and the final f value
#average the number of iterations and the final f value

#calculate f values for deconditioned A

#undue preconditioner







'''