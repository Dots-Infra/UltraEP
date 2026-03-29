"""
Deterministic round-robin reroute: expand logical routing to physical routing.

Reroute method maps a logical routing map
[num_tokens, num_global_logical_experts] to the physical expert space
[num_tokens, num_global_physical_experts], distributing tokens across replicas
in a deterministic round-robin fashion.

Two implementations are provided:
  1. CPU path (_RerouteProbsFunction):
     The C++ RerouteSolver computes index arrays on CPU, transfers them to GPU,
     and Python index operations scatter/gather the probabilities.
  2. CUDA path (_RerouteCUDAFunction):
     A fused CUDA kernel performs the round-robin assignment and probability
     scatter in a single GPU launch, avoiding all H2D/D2H transfers.

Gradient flow (both paths):
  Forward:  expanded_probs[t, phys] = probs[t, logical]   (scatter)
  Backward: grad_probs[t, logical]  = grad_out[t, phys]    (gather)
  The mapping between (t, logical) and (t, phys) is a 1-to-1 bijection
  over the N active routing pairs, so no duplicate-index issues arise.
"""

import torch


class _RerouteProbsFunction(torch.autograd.Function):
    """
    CPU-path autograd function for scattering probabilities from logical to physical
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


class _RerouteCUDAFunction(torch.autograd.Function):
    """
    CUDA-path autograd function for reroute with pre-allocated output buffers.

    The C++ Manager owns the output buffers and returns fresh ``from_blob``
    views (independent version counters) each call.  This eliminates the
    CUDA allocator overhead and memset launch latency visible in nsys.

    Forward buffer reuse across layers is correct when activation
    checkpointing is enabled (``moe_layer_recompute=True``), which is the
    standard MoE training configuration.  Backward buffer reuse is always
    correct because autograd processes layers sequentially in reverse.

    Non-differentiable arguments are stored as plain ctx attributes (not via
    ``save_for_backward``) to avoid version-counter issues from in-place
    updates to the shared placement buffer.
    """

    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,  # [T, L] float, requires_grad
        routing_map: torch.Tensor,  # [T, L] bool
        manager_runtime,  # _C.Manager (pybind11 object)
        layer_id: int,
    ):
        """
        Forward: scatter probs and construct expanded_routing_map via CUDA kernel
        using pre-allocated Manager buffers.

        Returns:
            expanded_probs: [T, P] float (differentiable)
            expanded_routing_map: [T, P] bool (non-differentiable)
        """
        expanded_probs, expanded_routing_map = manager_runtime.reroute_cuda_forward(
            layer_id, probs, routing_map
        )

        ctx.mark_non_differentiable(expanded_routing_map)

        # Save for backward (plain attributes — avoids version-counter checks)
        ctx.manager_runtime = manager_runtime
        ctx.routing_map = routing_map
        ctx.expanded_routing_map = expanded_routing_map
        ctx.layer_id = layer_id

        return expanded_probs, expanded_routing_map

    @staticmethod
    def backward(ctx, grad_expanded_probs: torch.Tensor, grad_expanded_routing_map):
        """
        Backward: gather gradients from physical to logical space via CUDA kernel
        using pre-allocated Manager buffers.
        """
        grad_probs = ctx.manager_runtime.reroute_cuda_backward(
            ctx.layer_id,
            grad_expanded_probs,
            ctx.routing_map,
            ctx.expanded_routing_map,
        )
        return grad_probs, None, None, None
