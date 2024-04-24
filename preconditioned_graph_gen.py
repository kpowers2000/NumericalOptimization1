import numpy as np
import matplotlib.pyplot as plt
import os
import time
from gradient_projection import paper_example_2, gradient_projection_method, moderated_spectral_preconditioner

# Define base directory and subdirectories
base_dir = 'final project graphs'
os.makedirs(base_dir, exist_ok=True)
subfolders = ['Condition_Number_vs_Iterations', 'Final_Function_Value_vs_Iterations', 'Average_Iterations_vs_n',
              'Distance_from_Initial_vs_Iterations', 'Iteration_Percent_Reduction', 'N_vs_Error']
for folder in subfolders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# Define values of n and scale values for preconditioning
n_values = [10, 20, 50, 100]
scale_values = np.arange(0.05, .50, 0.05)  # Scale values at 0.05 increments from 0.15 to 4.0
n_colors = ['blue', 'green', 'red', 'purple']  # Colors for each n value
scale_colors = plt.cm.viridis(np.linspace(0, 1, len(scale_values)))  # Use a colormap for varying scales

start_time_total = time.time()

# Initialize data storage for master graph
master_percent_reduction = {n: [] for n in n_values}
f_errors_per_scale = {scale: [] for scale in scale_values}

# Loop over different values of n
for n in n_values:
    start_time_n = time.time()
    iters_base = []
    error_base = []
    percent_reduction_per_scale = {scale: [] for scale in scale_values}

    # Run trials
    j_trials = 20
    for j in range(j_trials):
        A, b, lower_bound, upper_bound, x_initial, _ = paper_example_2(n)
        x_base, iter_base, f_base, fin_error_base = gradient_projection_method(n, A, b, upper_bound, lower_bound, x_initial, np.zeros(n))
        
        # Apply preconditioning for each scale and rerun
        for scale_idx, scale in enumerate(scale_values):
            M = moderated_spectral_preconditioner(A, scale)
            A_preconditioned = M @ A @ M
            x_pre, iter_pre, f_pre, fin_error_pre = gradient_projection_method(n, A_preconditioned, b, upper_bound, lower_bound, x_initial, x_base)
            percent_reduction = ((iter_base - iter_pre) / iter_base * 100) if iter_base > 0 else 0
            percent_reduction_per_scale[scale].append(percent_reduction)
            master_percent_reduction[n].append((scale, percent_reduction))
            f_error = abs(f_pre - f_base)
            f_errors_per_scale[scale].append((n, f_error))

    end_time_n = time.time()
    print(f'Time for n = {n}: {end_time_n - start_time_n:.2f} seconds')

# Plotting percent reduction in iterations per scale for each n
for n in n_values:
    plt.figure()
    for scale_idx, scale in enumerate(scale_values):
        reductions = percent_reduction_per_scale[scale]
        plt.scatter([scale] * len(reductions), reductions, color='blue')
    plt.title(f'Iteration Percent Reduction for n = {n}2')
    plt.xlabel('Scale Value')
    plt.ylabel('Percent Reduction in Iterations')
    plt.savefig(os.path.join(base_dir, 'Iteration_Percent_Reduction', f'Percent_Reduction_n{n}2.png'))
    plt.close()

# Master graph of all percent reductions
plt.figure()
for n_idx, n in enumerate(n_values):
    scales, reductions = zip(*master_percent_reduction[n])
    plt.scatter(scales, reductions, color=n_colors[n_idx], label=f'n = {n}')
plt.title('Percent Reduction in Iterations for all n values')
plt.xlabel('Scale Value')
plt.ylabel('Percent Reduction in Iterations')
plt.legend()
plt.savefig(os.path.join(base_dir, 'Iteration_Percent_Reduction', 'Master_Percent_Reduction3.png'))
plt.close()

# Plotting n vs f_error for all scales
plt.figure()
for scale_idx, scale in enumerate(scale_values):
    ns, errors = zip(*f_errors_per_scale[scale])
    plt.scatter(ns, errors, color=scale_colors[scale_idx], label=f'Scale {scale:.2f}')
plt.title('n vs Final Error in Objective Function Across All Trials3')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Error in Objective Function')
plt.legend()
plt.savefig(os.path.join(base_dir, 'N_vs_Error', 'N_vs_F_Error3.png'))
plt.close()

end_time_total = time.time()
total_time = end_time_total - start_time_total
print(f'Total execution time: {total_time:.2f} seconds')
