import torch

def batch_lookup_tensors( #double chunked
    batch_pairs: list,          # list of (u,e)
    samples_db: torch.Tensor,   # [N,16]
    tree_outputs: torch.Tensor, # [N,O]
    q_chunk_size: int = 128,
    db_chunk_size: int = 50_000
) -> torch.Tensor:              # returns [B,O]
    device = samples_db.device
    dtype  = samples_db.dtype

    B = len(batch_pairs)
    O = tree_outputs.size(1)
    N = samples_db.size(0)

    # Build [B,16] query
    q_u = torch.stack([torch.tensor(u, dtype=dtype, device=device) for u,_ in batch_pairs], dim=0)
    q_e = torch.stack([torch.tensor(e, dtype=dtype, device=device) for _,e in batch_pairs], dim=0)
    q_full = torch.cat([q_u, q_e], dim=1).float()  # [B,16]

    # Prepare output + running best distances/indices
    results    = torch.empty((B, O), dtype=tree_outputs.dtype, device=device)
    best_d2    = torch.full((B,), float('inf'), device=device)
    best_index = torch.zeros((B,), dtype=torch.long, device=device)

    # Sweep through database in chunks
    for db_start in range(0, N, db_chunk_size):
        db_end   = min(db_start + db_chunk_size, N)
        db_chunk = samples_db[db_start:db_end].float()      # [D,16]
        out_chunk= tree_outputs[db_start:db_end]            # [D,O]

        # For each query‐chunk
        for q_start in range(0, B, q_chunk_size):
            q_end    = min(q_start + q_chunk_size, B)
            q_chunk  = q_full[q_start:q_end]                # [b,16]

            # compute [b, D] d2 slice
            # (D×16 floats vs b×D×16 floats, small enough if D~50k, b~128)
            diffs = db_chunk.unsqueeze(0) - q_chunk.unsqueeze(1)  # [b,D,16]
            d2    = (diffs*diffs).sum(dim=2)                      # [b,D]

            # compare to running best
            # broadcast best_d2[q_start:q_end] → [b,1]
            mask = d2 < best_d2[q_start:q_end].unsqueeze(1)
            # update best_d2, best_index
            best_d2[q_start:q_end] = torch.where(mask.any(1),
                                                 torch.min(d2, dim=1).values,
                                                 best_d2[q_start:q_end])
            # find new winner indices within this chunk
            new_idxs = torch.where(mask.any(1),
                                   d2.argmin(dim=1) + db_start,
                                   best_index[q_start:q_end])
            best_index[q_start:q_end] = new_idxs

    # Now best_index[i] is the index in 0..N of the nearest neighbor for query i
    results = tree_outputs[best_index]  # [B,O]
    return results


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
