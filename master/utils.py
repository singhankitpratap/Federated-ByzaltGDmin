import torch
from torch.linalg import qr, eigh
torch.set_default_dtype(torch.float64)
def orthonormalize(U):
    U = U.to(torch.float64)
    Q, R = qr(U)
    return Q

def compute_projection_matrix(U):
    U = U.to(torch.float64)
    return U @ U.T

def weiszfeld_algorithm(A, T_gm, epsilon_gm):
    A = torch.stack(A).to(torch.float64)
    L, d = A.shape
    z_t = A.mean(dim=0)

    for t in range(T_gm):
        numer = torch.zeros_like(z_t, dtype=torch.float64)
        denom = torch.tensor(0.0, dtype=torch.float64)

        for i in range(L):
            dist = torch.norm(A[i] - z_t)
            if dist == 0:
                dist = 1.0
            weight = 1.0 / dist
            numer += weight * A[i]
            denom += weight

        z_t1 = numer / denom

        if torch.norm(z_t1 - z_t) < epsilon_gm:
            break
        z_t = z_t1

    return z_t


def subspace_median(U_list, T_gm, epsilon_gm):
    U_list = [orthonormalize(U).to(torch.float64)  for U in U_list]
    P_list = [compute_projection_matrix(U).to(torch.float64) for U in U_list]
    P_list_flattened = [P.view(-1).to(torch.float64) for P in P_list]
    P_gm_flat = weiszfeld_algorithm(P_list_flattened, T_gm, epsilon_gm)
    P_gm = P_gm_flat.view(P_list[0].shape)
    l_best = torch.argmin(torch.tensor([torch.norm(P - P_gm) for P in P_list], dtype=torch.float64))
    return U_list[l_best]

