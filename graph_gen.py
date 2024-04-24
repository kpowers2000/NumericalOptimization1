import numpy as np
import matplotlib.pyplot as plt
import os
import time
from gradient_projection import paper_example_2, gradient_projection_method

# Define the base directory for saving graphs
base_dir = 'final project graphs'
os.makedirs(base_dir, exist_ok=True)

# Define subdirectories for each type of graph
subfolders = ['Condition_Number_vs_Iterations', 'Final_Function_Value_vs_Iterations', 'Average_Iterations_vs_n', 
              'Distance_from_Initial_vs_Iterations', 'Master_Graphs']
for folder in subfolders:
    os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

# Plot master graphs without major outliers
def remove_outliers(data):
    # Extract iteration values and filter based on these
    iter_values = [x[1] for x in data]  # Extract iteration values for outlier detection
    q1 = np.percentile(iter_values, 25)
    q3 = np.percentile(iter_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if x[1] >= lower_bound and x[1] <= upper_bound]

def remove_extreme_outliers(data, threshold=3):
    # Calculate mean and standard deviation of the numerical data
    mean_iters = np.mean(data)
    std_iters = np.std(data)
    # Filter data to remove points that are more than `threshold` standard deviations from the mean
    return [x for x in data if abs(x - mean_iters) <= threshold * std_iters]
def remove_outliers_2(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in data if x >= lower_bound and x <= upper_bound]
# Define different values of n to iterate over
n_values = [10, 20, 50, 100]
avg_iters = []
std_dev_iters = []
n_time_values = []
n_colors = ['blue', 'green', 'red', 'purple']  # Colors for each n value

# Start total time measurement
start_time_total = time.time()

# Master data storage for master graphs
master_cond_vs_iters = []
master_dist_vs_iters = []
iterations_data = {n: [] for n in n_values}

# Loop over different values of n
for n_index, n in enumerate(n_values):
    # Start time for this value of n
    start_time_n = time.time()

    # Initialize arrays to store results for each n
    f_finals = []
    iters_array = []
    cond_num_array = []
    x_distance_from_initial = []
    A_condition_numbers = []
    iterations_per_trial = []

    # Number of trials
    j_trials = 20

    # Loop through the number of trials for each n
    for j in range(j_trials):
        A, b, lower_bound, upper_bound, x_initial, _ = paper_example_2(n)
        cond_num = np.linalg.cond(A)
        cond_num_array.append(cond_num)
        A_condition_numbers.append(cond_num / 1e6)  # Divide by a million for scaling
        x, iter_val, f_final, _ = gradient_projection_method(n, A, b, upper_bound, lower_bound, x_initial, np.zeros(n))
        f_finals.append(f_final)
        iters_array.append(iter_val)
        x_distance_from_initial.append(np.linalg.norm(x - x_initial))
        # Collect data for master graphs
        # master_cond_vs_iters.append((cond_num / 1e6, iter_val, n_colors[n_index]))
        # master_dist_vs_iters.append((np.linalg.norm(x - x_initial), iter_val, n_colors[n_index]))
        iterations_data[n].append(iter_val)


    # End time for this value of n
    end_time_n = time.time()
    print(f'Time for n = {n}: {end_time_n - start_time_n:.2f} seconds')
    n_time_values.append(end_time_n - start_time_n)


# plt.figure()
# filtered_data = remove_outliers(master_cond_vs_iters)
# for cond_num, iters, color in filtered_data:
#     plt.scatter(cond_num, iters, color=color)
# plt.title('Condition Number vs Iterations for all n (No Major Outliers)')
# plt.xlabel('Condition Number (in millions)')
# plt.ylabel('Number of Iterations')
# plt.savefig(os.path.join(base_dir, 'Master_Graphs', 'Condition_Number_vs_Iterations_for_all_n_no_outliers.png'))
# plt.close()

# plt.figure()
# filtered_data = remove_outliers(master_dist_vs_iters)
# for dist, iters, color in filtered_data:
#     plt.scatter(dist, iters, color=color)
# plt.title('Distance from Initial vs Iterations for all n (No Major Outliers)')
# plt.xlabel('Distance from Initial')
# plt.ylabel('Number of Iterations')
# plt.savefig(os.path.join(base_dir, 'Master_Graphs', 'Distance_from_Initial_vs_Iterations_for_all_n_no_outliers.png'))
# plt.close()

# Plot average iterations with error bars
filtered_avg_iters = []
filtered_std_dev_iters = []
for n in n_values:
    #filtered_iters = remove_outliers_2(iterations_data[n])
    filtered_iters = iterations_data[n]
    avg_iters = np.mean(filtered_iters)
    std_dev_iters = np.std(filtered_iters)
    filtered_avg_iters.append(avg_iters)
    filtered_std_dev_iters.append(std_dev_iters)

# Plot average iterations with error bars using filtered data
plt.figure()
for idx, n in enumerate(n_values):
    plt.errorbar(n, filtered_avg_iters[idx], yerr=filtered_std_dev_iters[idx], fmt='-o', capsize=5, color=n_colors[idx], label=f'n = {n}')
plt.title('Average Number of Iterations vs Matrix Size')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Average Number of Iterations')
plt.legend()
plt.savefig(os.path.join(base_dir, 'Average_Iterations_vs_n', 'Avg_Iterations_vs_Matrix_Size_no_outliers.png'))
plt.close()

# End total time measurement
end_time_total = time.time()
total_time = end_time_total - start_time_total
# Loop through and print execution time for each n
for i in range(len(n_values)):
    print(f'Time for n = {n_values[i]}: {n_time_values[i]:.2f} seconds')
print(f'Total execution time: {total_time:.2f} seconds')
