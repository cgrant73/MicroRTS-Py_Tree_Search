import torch
#a set of two scalars, a hashmap of treeoutpus, and a hashmap samples_db

tree_outputs = torch.load('tree_outputs.pt')
print("Loaded tree_outputs:", tree_outputs.shape) # shape [1000000,16]
samples_db = torch.load('samples_db.pt')  # shape [1000000,16]
print("Loaded samples_db:", samples_db.shape)

user_state = [0, 0, 0, 0, 0, 0, 0, 0]
enemy_state = [1, 1, 1, 1, 1, 1, 1, 1]

q = torch.tensor(user_state + enemy_state, dtype=samples_db.dtype)  # [16]

# 3) Compute all squared distances in one vectorized op
#    cast to float if you like (optional)
db = samples_db.float()                        # [N,16]
q  = q .float()                                # [16]
# Broadcast subtraction → [N,16], then square & sum → [N]
d2 = (db - q).pow(2).sum(dim=1)                # [N]

# 4) Grab the index of the nearest neighbor
idx = torch.argmin(d2).item()

# 5) Lookup its precomputed output
closest_out = tree_outputs[idx]                # [O], e.g. [21]



print(f"Closest index = {idx}, distance = {d2[idx].item():.2f}")
print("Precomputed output for that pair:", closest_out)
