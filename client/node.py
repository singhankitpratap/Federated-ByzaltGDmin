import torch
from utils import indicator_function
torch.set_default_dtype(torch.float64)

# Node Algorithm Functions
def Byz_altGDmin_threshold(Y_l, kappa, mu, m, q):
    alpha_l = (9 * kappa**2 * mu**2 / (m * q)) * torch.sum(torch.norm(Y_l, dim=0)**2)
    return alpha_l

def Byz_altGDmin_initialization(Y_l, A_l, alpha, r, q):
    Xhat_0_l = torch.zeros(A_l[0].shape[1], q)
    for k in range(q):
        y_truncated = Y_l[:, k] * indicator_function(Y_l[:, k].abs() <= alpha**0.5)
        e_k = torch.eye(q, dtype=torch.float64)[:, k].view(-1, 1)
        Xhat_0_l += torch.matmul(A_l[k].t(), y_truncated.unsqueeze(1)) @ e_k.T
    U_0_l = torch.linalg.svd(Xhat_0_l, full_matrices=False)[0][:, :r]
    return U_0_l

def Byz_altGDmin_GD_iterations(Y_l, A_l, U_t_minus_1, q):
    U = U_t_minus_1
    b_k_l = []
    for k in range(q):
        b_k = torch.linalg.pinv(torch.matmul(A_l[k], U)).matmul(Y_l[:, k].unsqueeze(1)).to(torch.float64)
        b_k_l.append(b_k)
    grad_f_l = torch.zeros_like(U, dtype=torch.float64)
    for k in range(q):
        A_k_U_b_k = torch.matmul(A_l[k], U).matmul(b_k_l[k])  # Shape (m_L, 1)
        error = A_k_U_b_k - Y_l[:, k].unsqueeze(1)  # Shape (m_L, 1)
        grad_component = torch.matmul(A_l[k].t(), error)  # Shape (n, 1)
        grad_f_l += grad_component.matmul(b_k_l[k].t())  # Shape (n, r)
    return grad_f_l


