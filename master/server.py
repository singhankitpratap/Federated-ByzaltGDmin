import torch
import numpy as np
import random
from central_server.utils import subspace_median, weiszfeld_algorithm
torch.set_default_dtype(torch.float64)

# Function to append error values to a file
def log_error(filename, iteration, error):
    with open(filename, 'a') as f:
        f.write(f"{iteration},{error}\n")



# Algorithm 1: Byz-altGDmin: Threshold Server
def Byz_altGDmin_threshold_server(alpha_l_list, L_byz, L):
    alpha_mean = -10 / L * sum(alpha_l_list)
    selected_indices = random.sample(range(L), L_byz)
    for idx in selected_indices:
        alpha_l_list[idx] = alpha_mean
    alpha = np.median(alpha_l_list)
    return alpha

# Algorithm 2: Byz-altGDmin: Initialization Server
def Byz_altGDmin_initialization_server(U_0_list, L_byz, L, n, r, U_star, T_gm, epsilon_gm):
    alt_matrix = torch.tensor([10 if (i + j) % 2 == 0 else -10 for i in range(n) for j in range(r)], dtype=torch.float64).reshape(n, r)
    selected_indices = random.sample(range(L), L_byz)
    for idx in selected_indices:
        U_0_list[idx] = alt_matrix.float()
    U_out = subspace_median(U_0_list, T_gm, epsilon_gm)
    U_0 = U_out
    error_init = torch.norm((torch.eye(n, dtype=torch.float64) - U_star @ U_star.t()) @ U_0)
    print(f"Initialization Error: {error_init}")
    return U_0

filename = "error_log.txt"
# Algorithm 3: Byz-altGDmin: GD Server
def Byz_altGDmin_GD_server(gradients, L_byz, L, eta, m, U_t_minus_1, U_star, T_gm, epsilon_gm, iteration, r):

    grad_mean = -10 / L * sum(gradients)
    selected_indices = random.sample(range(L), L_byz)
    for idx in selected_indices:
        gradients[idx] = grad_mean.to(torch.float64)
    grad_flattened = [grad.reshape(-1).to(torch.float64) for grad in gradients]
    grad_gm_flat = weiszfeld_algorithm(grad_flattened, T_gm, epsilon_gm)
    grad_gm = grad_gm_flat.view(gradients[0].shape).to(torch.float64)
    U_plus = torch.linalg.qr(U_t_minus_1 - eta  * grad_gm)[0]
    U_t = U_plus
    error_gd = torch.norm((torch.eye(U_t.size(0), dtype=torch.float64) - U_star @ U_star.t()) @ U_t) / torch.sqrt(torch.tensor(r, dtype=torch.float64))
    print(f"SD: {error_gd}")
    log_error(filename, iteration, error_gd)
    return U_t, error_gd

