import numpy as np
import os

METHOD_TO_TRY = "StructurePreservingPODReductor"
folder_name_plot = f"src/pymordemos/structure_preserving_POD/plot_data/plot_data_{METHOD_TO_TRY}"
folder_name_results = f"src/pymordemos/structure_preserving_POD/results/results_{METHOD_TO_TRY}"

with open(os.path.join(folder_name_results, "methods_used"), 'r') as f:
    METHODS = [line.strip() for line in f]

time = np.loadtxt(os.path.join(folder_name_results, f"time.txt"))

fom_Hamiltonian = np.loadtxt(os.path.join(folder_name_results, "Hamiltonian_fom"))
fom_Hamiltonian_data = np.column_stack((time[:998], fom_Hamiltonian))
np.savetxt(os.path.join(folder_name_plot, "fom_Hamiltonian"), fom_Hamiltonian_data)

x_axis = np.arange(0, 1, 0.002)

snapshot_matrix = np.loadtxt(os.path.join(folder_name_results, f"solution_trajectory"))
snapshot_matrix_data_0 = np.column_stack((x_axis, snapshot_matrix[:, 0]))
np.savetxt(os.path.join(folder_name_plot, f"solution_trajectory_t_0"), snapshot_matrix_data_0)


for method in METHODS:
    red_dims_used = np.loadtxt(os.path.join(folder_name_results, f"red_dims_used_{method}"))
    for dim in red_dims_used:
        H = np.loadtxt(os.path.join(folder_name_results, f"Hamiltonian_reconstruction_{method}_{int(dim)}.txt"))
        Hamiltonian_data = np.column_stack((time[:998], H))
        np.savetxt(os.path.join(folder_name_plot, f"Hamiltonian_plot_{method}_{int(dim)}.txt"), Hamiltonian_data)

    reconstruction_dim = red_dims_used.max()
    reconstruction = np.loadtxt(os.path.join(folder_name_results, f"reconstruction_q_{method}_{int(reconstruction_dim)}.txt"))
    
    time_indices = [249, 499]
    stop = 500
    step = 60
    for time_index in time_indices:
        if method == 'POD':
            slice = np.arange(start=15, stop=stop, step=step)
        elif method == 'POD_PH':
            slice = np.arange(start=30, stop=stop, step=step)
        elif method == "extended_snapshot_POD":
            slice = np.arange(start=45, stop=stop, step=step)
        else:
            slice = np.arange(stop=stop, step=step)
            slice = np.append(slice, 499)
        t = round(time[time_index],1)
        reconstruction_data_sliced = np.column_stack((x_axis[slice], reconstruction[:, time_index][slice]))
        np.savetxt(os.path.join(folder_name_plot, f"reconstruction_{method}_{int(reconstruction_dim)}_t_{t}"), reconstruction_data_sliced)

        reconstruction_data = np.column_stack((x_axis, reconstruction[:, time_index]))
        np.savetxt(os.path.join(folder_name_plot, f"reconstruction_{method}_{int(reconstruction_dim)}_unsliced_t_{t}"), reconstruction_data)

        snapshot_matrix_data = np.column_stack((x_axis, snapshot_matrix[:, time_index]))
        np.savetxt(os.path.join(folder_name_plot, f"solution_trajectory_t_{t}"), snapshot_matrix_data)

        
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





