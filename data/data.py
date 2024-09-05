import torch
import json
torch.set_default_dtype(torch.float64)



# Set parameters
n = 600
q = 600
r = 4
m = 198
L = 6
m_L = m // L
L_byz_values = [1, 2]

# Step 1: Generate U_star by orthogonalizing an n x r standard Gaussian matrix
U_star = torch.randn(n, r, dtype=torch.float64)

U_star, _ = torch.linalg.qr(U_star)

# Step 2: Generate columns of B_star from N(0, I_r)
B_star = torch.randn(r, q, dtype=torch.float64)


# Step 3: Set X_star = U_star * B_star
X_star = torch.matmul(U_star, B_star)

# Step 4: Generate A_k matrices of size m x n with i.i.d. standard Gaussian entries
A_k = [torch.randn(m, n, dtype=torch.float64) for _ in range(q)]


# Step 5: Set y_k = A_k * x_star_k for all k in [q]
y_k = [torch.matmul(A_k[k], X_star[:, k]) for k in range(q)]

# Step 6: Distribute data to nodes
data_nodes = {}
for l in range(L):
    Y_l = torch.stack([y_k[k][l*m_L:(l+1)*m_L] for k in range(q)], dim=1)
    A_l = [A_k[k][l*m_L:(l+1)*m_L, :] for k in range(q)]
    data_nodes[l] = {'Y_l': Y_l, 'A_l': A_l}

_, S, _ = torch.svd(X_star)
kappa = S[0] / S[r-1]
column_norms = torch.norm(X_star, dim=0)

# Square the norms
column_norms_squared = column_norms ** 2

# Find the maximum value among the squared norms
max_column_norm_squared = torch.max(column_norms_squared)

mu = torch.sqrt((max_column_norm_squared*q)/ (S[0]**2*r))

# Example scalar values
scalars = {
    "kappa": kappa.item(),
    "mu": mu.item(),
    "sigma_max": S[0].item(),
    "q": q,
    "r": r,
    "L": L,
    "m": m,
    "n": n,
    "m_L": m_L,
}

# Save to JSON file
with open('scalars.json', 'w') as json_file:
    json.dump(scalars, json_file)

# Save data for each node
torch.save(data_nodes, 'data_nodes.pth')

torch.save(U_star, 'U_star.pth')

data_nodes = torch.load('data_nodes.pth')
for key, data in data_nodes.items():
    torch.save(data, f'data_node_{key}.pth')