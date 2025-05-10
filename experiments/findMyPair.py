import torch

def batch_lookup_tensors(
    user_state: torch.Tensor,    # [B,8]
    enemy_state: torch.Tensor,   # [B,8]
    samples_db: torch.Tensor,    # [N,16]
    tree_outputs: torch.Tensor,  # [N,O]
    q_chunk_size: int = 128,
    db_chunk_size: int = 50_000
) -> torch.Tensor:
    device = samples_db.device
    dtype  = samples_db.dtype

    B = user_state.size(0)
    O = tree_outputs.size(1)
    N = samples_db.size(0)

    #prep
    q_u = user_state.to(device=device, dtype=dtype).float()
    q_e = enemy_state.to(device=device, dtype=dtype).float()
    q_full = torch.cat([q_u, q_e], dim=1)  # [B,16]

    #init
    best_d2 = torch.full((B,), float('inf'), device=device)
    best_index = torch.zeros((B,), dtype=torch.long, device=device)

    #sweeping databse
    for db_start in range(0, N, db_chunk_size):
        db_end = min(db_start + db_chunk_size, N)
        db_chunk = samples_db[db_start:db_end].float()  # [D,16]
        out_chunk = tree_outputs[db_start:db_end]       # [D,O]

        #sweeping queries
        for q_start in range(0, B, q_chunk_size):
            q_end = min(q_start + q_chunk_size, B)
            q_chunk = q_full[q_start:q_end]  # [b,16]

            # Compute squared distances [b,D]
            diffs = db_chunk.unsqueeze(0) - q_chunk.unsqueeze(1)  # [b,D,16]
            d2 = (diffs * diffs).sum(dim=2)                       # [b,D]

            # Compare to current best
            current_best = best_d2[q_start:q_end].unsqueeze(1)   # [b,1]
            mask = d2 < current_best                            # [b,D]

            # Compute minima in this slice
            min_d2, min_idx = d2.min(dim=1)
            update_mask = mask.any(dim=1)

            # Update best distances and indices
            best_d2[q_start:q_end][update_mask] = min_d2[update_mask]
            best_index[q_start:q_end][update_mask] = min_idx[update_mask] + db_start
            
    results = tree_outputs[best_index]  # [B,O]
    return results


# if __name__ == "__main__":
#     # Simulated data
#     N, O = 10, 21
#     samples_db = torch.randn(N, 16)
#     tree_outputs = torch.randint(0, 2, (N, O), dtype=torch.float32)

#     # Example batch of user/enemy states
#     user_state = torch.tensor([
#         [0,1,2,3,4,1,1,0],
#         [5,5,5,5,5,1,1,5],
#         [2,3,1,0,4,1,1,2],
#     ], dtype=torch.float32)
#     enemy_state = torch.tensor([
#         [5,4,3,2,1,0,0,5],
#         [0,0,0,0,0,0,0,0],
#         [2,3,1,0,4,1,1,2],
#     ], dtype=torch.float32)

#     out = batch_lookup_tensors(user_state, enemy_state, samples_db, tree_outputs)
#     print("Output shape:", out.shape)  # should be [3,21]
#     print(out)