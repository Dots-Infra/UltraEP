"""Autograd wrapper for dense reroute on device."""

import torch


class _DenseRerouteFunction(torch.autograd.Function):
    """Dense reroute autograd bridge backed by the C++ runtime."""

    @staticmethod
    def forward(
        ctx,
        probs: torch.Tensor,  # [T, L] float, requires_grad
        routing_map: torch.Tensor,  # [T, L] bool
        manager,
        layer_id: int,
    ):
        expanded_probs, expanded_routing_map = manager.runtime.dense_reroute_forward(
            layer_id, probs, routing_map
        )

        ctx.mark_non_differentiable(expanded_routing_map)

        # Keep the Python allocation owner alive until backward has finished.
        ctx.manager = manager
        ctx.routing_map = routing_map
        ctx.expanded_routing_map = expanded_routing_map
        ctx.layer_id = layer_id

        return expanded_probs, expanded_routing_map

    @staticmethod
    def backward(ctx, grad_expanded_probs: torch.Tensor, grad_expanded_routing_map):
        grad_probs = ctx.manager.runtime.dense_reroute_backward(
            ctx.layer_id,
            grad_expanded_probs,
            ctx.routing_map,
            ctx.expanded_routing_map,
        )
        return grad_probs, None, None, None
