import numpy as np
import os


folder_name_plot = "src/pymordemos/structure_preserving_POD/plot_data"
folder_name_results = "src/pymordemos/structure_preserving_POD/results"

with open(os.path.join(folder_name_results, "methods_used"), 'r') as f:
    METHODS = [line.strip() for line in f]

for method in METHODS:
    red_dims_used = np.loadtxt(os.path.join(folder_name_results, f"red_dims_used_{method}"))
    for dim in red_dims_used:
        H = np.loadtxt(os.path.join(folder_name_results, f"Hamiltonian_reconstruction_{method}_{int(dim)}.txt"))
        time = np.loadtxt(os.path.join(folder_name_results, f"time.txt"))
        Hamiltonian_data = np.column_stack((time, H))
        np.savetxt(os.path.join(folder_name_plot, f"Hamiltonian_plot_{method}_{int(dim)}.txt"), Hamiltonian_data)

    reconstruction_dim = red_dims_used.max()
    reconstruction = np.loadtxt(os.path.join(folder_name_results, f"reconstruction_q_{method}_{int(reconstruction_dim)}.txt"))
    x_axis = np.arange(0, 1, 0.001)
    slice = np.arange(stop=500, step=25)
    time_index = 998//2 + 1 
    reconstruction_data = np.column_stack((x_axis[slice], reconstruction[:, time_index][slice]))
    np.savetxt(os.path.join(folder_name_plot, f"reconstruction_{method}_{int(reconstruction_dim)}_t_{time[time_index]}"), reconstruction_data)

        
    relative_reduction_error = np.loadtxt(os.path.join(folder_name_results, f"relative_reduction_error_{method}"))
    relative_projection_error = np.loadtxt(os.path.join(folder_name_results, f"relative_projection_error_{method}"))
    for i in range(relative_reduction_error.shape[0]):
        if np.isnan(relative_reduction_error[i]):
            relative_reduction_error = relative_reduction_error[:i]
            relative_projection_error = relative_projection_error[:i]
            break
    reduction_data = np.column_stack((red_dims_used, relative_reduction_error))
    np.savetxt(os.path.join(folder_name_plot, f"relative_reduction_error_plot_{method}"), reduction_data)

    
    projection_data = np.column_stack((red_dims_used, relative_projection_error))
    np.savetxt(os.path.join(folder_name_plot, f"relative_projection_error_plot_{method}"), projection_data)



