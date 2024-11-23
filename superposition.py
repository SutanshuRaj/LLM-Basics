import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

mps_device = torch.device("mps")

# Number of Vectors >>> Number of Matrix Dimensions
num_vectors = 10000
vector_len = 100

matrix_dim = torch.randn(num_vectors, vector_len, device=mps_device)
matrix_dim /= matrix_dim.norm(p=2, dim=1, keepdim=True)
matrix_dim.requires_grad_(True)

# Optimization to create Perpendicular Vectors.
optimizer = torch.optim.Adam([matrix_dim], lr=0.01)
iters = 128

losses = []
dot_diff_cutoff = 0.01
matrix_id = torch.eye(num_vectors, num_vectors, device=mps_device)

for step in tqdm(range(iters)):
	optimizer.zero_grad()

	dot_products = matrix_dim @ matrix_dim.T

	# Penalty for Deviation.
	diff = dot_products - matrix_id
	loss = (diff.abs() - dot_diff_cutoff).relu().sum()

	# Incentive to keep rows Normalized.
	loss += num_vectors * diff.diag().pow(2).sum()

	loss.backward()
	optimizer.step()
	losses.append(loss.item())


# Iteration-Loss Curve.
plt.plot(losses)
plt.grid(True)
plt.show()

# Angle Distribution Plot.
plot=False
if plot:
	dot_products = matrix_dim @ matrix_dim.T
	norms = torch.sqrt(torch.diag(dot_products))
	norm_dot_product = dot_products / torch.outer(norms, norms)
	angle_degrees = torch.rad2deg(torch.acos(norm_dot_product.detach()))

	self_orthogonality_mask = ~(torch.eye(num_vectors, num_vectors).bool())

	plt.hist(angle_degrees[self_orthogonality_mask].cpu().numpy().ravel(), bins=1000, range=(0, 180))
	plt.grid(True)
	plt.show()