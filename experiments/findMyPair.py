import torch

def batch_lookup_tensors(
    batch_pairs: list,      # [B,8]
    samples_db: torch.Tensor, # [N,16]
    tree_outputs: torch.Tensor# [N,O]
) -> torch.Tensor:            # returns [B,O]

        # Convert list of pairs into two [B,8] tensors
    q_u = torch.stack([torch.tensor(u, dtype=samples_db.dtype) for u, _ in batch_pairs], dim=0)
    q_e = torch.stack([torch.tensor(e, dtype=samples_db.dtype) for _, e in batch_pairs], dim=0)

    # 1) build B×16 query
    q = torch.cat([q_u, q_e], dim=1).float()       # [B,16]
    # 2) compute distances
    db = samples_db.float()                        # [N,16]
    diffs = db.unsqueeze(0) - q.unsqueeze(1)       # [B,N,16]
    d2    = (diffs*diffs).sum(dim=2)               # [B,N]
    # 3) find nearest and gather
    idxs = d2.argmin(dim=1)                        # [B]
    return tree_outputs[idxs]                     # [B,O]

# if __name__ == "__main__":
#     # Simulated data
#     N, O = 10, 21
#     samples_db = torch.randn(N, 16)
#     tree_outputs = torch.randint(0, 2, (N, O), dtype=torch.float32)

#     # Batch of 3 random queries
#     batch_pairs = [
#         ([0,1,2,3,4,1,1,0], [5,4,3,2,1,0,0,5]),
#         ([5,5,5,5,5,1,1,5], [0,0,0,0,0,0,0,0]),
#         ([2,3,1,0,4,1,1,2], [2,3,1,0,4,1,1,2]),
#     ]

#     out = batch_lookup_tensors(batch_pairs, samples_db, tree_outputs)
#     for row in out:
#         print(row)
        
#     print("Output shape:", out.shape)  # should be [3,21]
