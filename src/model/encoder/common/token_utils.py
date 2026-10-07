import torch


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
    query = token_tensor[:, 0:1, ...].expand(batch_size, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:, ...].expand(batch_size, sequence_length - 1, *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    return combined.view(batch_size * sequence_length, *combined.shape[2:])
