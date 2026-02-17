"""
Deterministic round-robin reroute: expand logical routing to physical routing.

Reroute method maps a logical routing map
[num_tokens, num_global_logical_experts] to the physical expert space
[num_tokens, num_global_physical_experts], distributing tokens across replicas
in a deterministic round-robin fashion.

The heavy-lifting (round-robin index computation) is done by the C++ RerouteSolver.
This module adds the torch.autograd.Function wrapper so that gradients flow
correctly through the probability tensor during training.

Gradient flow:
  Forward:  expanded_probs[t, phys] = probs[t, logical]   (scatter)
  Backward: grad_probs[t, logical]  = grad_out[t, phys]    (gather)
  The mapping between (t, logical) and (t, phys) is a 1-to-1 bijection
  over the N active routing pairs, so no duplicate-index issues arise.
"""

import torch


class _RerouteProbsFunction(torch.autograd.Function):
    """
    Custom autograd function for scattering probabilities from logical to physical
    expert space, and gathering gradients back.

    The mapping (token_idx, logical_idx) <-> (token_idx, physical_idx) is pre-computed
    by the C++ RerouteSolver and passed as non-differentiable arguments.
    Only `probs` carries gradients.
    """

    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,  # [T, L] float, requires_grad
        token_idx: torch.Tensor,  # [N] int64
        logical_idx: torch.Tensor,  # [N] int64
        physical_idx: torch.Tensor,  # [N] int64
        num_physical: int,
    ) -> torch.Tensor:
        """
        Scatter probs from logical to physical space.

        Returns:
            expanded_probs: [T, P] float (same dtype as probs)
        """
        T = probs.size(0)
        P = num_physical

        # Save mapping for backward (int tensors, no grad needed)
        ctx.save_for_backward(token_idx, logical_idx, physical_idx)
        ctx.T = T
        ctx.L = probs.size(1)

        # Scatter: expanded_probs[t, phys] = probs[t, logical]
        expanded_probs = torch.zeros(T, P, dtype=probs.dtype, device=probs.device)
        if token_idx.numel() > 0:
            expanded_probs[token_idx, physical_idx] = probs[token_idx, logical_idx]

        return expanded_probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Gather gradients from physical space back to logical space.

        Returns:
            grad_probs: [T, L] float, plus None for non-differentiable args.
        """
        token_idx, logical_idx, physical_idx = ctx.saved_tensors

        # Gather: grad_probs[t, logical] = grad_output[t, phys]
        grad_probs = torch.zeros(
            ctx.T, ctx.L, dtype=grad_output.dtype, device=grad_output.device
        )
        if token_idx.numel() > 0:
            grad_probs[token_idx, logical_idx] = grad_output[token_idx, physical_idx]

        return grad_probs, None, None, None, None
