# -*- coding: utf-8 -*-
"""Drop-in replacement for torch.utils.checkpoint.checkpoint that keeps
BatchNorm running statistics from being accumulated twice per iteration.

See pytorch/pytorch#96136. Gradient checkpointing runs the wrapped function
twice per training iteration (once under no_grad to produce the output, once
with grad enabled to rebuild the graph). Both passes update the running
statistics of any BatchNorm inside the wrapped region, so the same batch is
accumulated twice — algebraically equivalent to raising BN's momentum from
``m`` to ``2m - m**2`` (0.1 -> 0.19 at the PyTorch default).

This module protects the buffers during the *recomputation* pass, so the
statistics recorded are the ones from the pass whose output actually feeds
the rest of the network — matching what a non-checkpointed run records.
"""
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

_TRACKED = ('running_mean', 'running_var', 'num_batches_tracked')


def _protect(function, module):
    """Wrap ``function`` so BN buffers under ``module`` survive recomputation.

    A fresh wrapper (and therefore a fresh ``state``) is built for every
    ``checkpoint`` call, so the flag tracks the two passes of that one call.
    Counting passes rather than testing ``torch.is_grad_enabled()`` is what
    makes this correct under ``use_reentrant=False`` too: there both passes
    run with grad enabled, so a grad-mode test would suppress both updates.
    """
    state = {'first_pass_done': False}

    def wrapper(*args, **kwargs):
        # First pass: let BN update exactly as it would without checkpointing.
        if not state['first_pass_done']:
            state['first_pass_done'] = True
            return function(*args, **kwargs)
        # Recomputation pass: rebind the buffers to clones, restore afterwards.
        saved = []
        try:
            for sub in module.modules():
                if isinstance(sub, _BatchNorm):
                    for name in _TRACKED:
                        value = getattr(sub, name, None)
                        if value is not None:
                            saved.append((sub, name, value))
                            setattr(sub, name, value.detach().clone())
            return function(*args, **kwargs)
        finally:
            for sub, name, value in reversed(saved):
                setattr(sub, name, value)

    return wrapper


def checkpoint(function, *args, module=None, **kwargs):
    """``torch.utils.checkpoint.checkpoint`` with BN statistics protection.

    Args:
        function (callable): the function to checkpoint, as usual.
        module (nn.Module, optional): the module owning the BatchNorm layers
            inside ``function``. Pass ``self`` from the calling block. When
            omitted the call falls through to the stock implementation, so
            this is a safe drop-in.

    Returns:
        Same as ``torch.utils.checkpoint.checkpoint``.
    """
    if module is not None and module.training:
        function = _protect(function, module)
    return cp.checkpoint(function, *args, **kwargs)


def find_contaminated_batchnorms(model):
    """Return BN layers whose ``num_batches_tracked`` advances more than once.

    Run one full training iteration between the two calls::

        before = snapshot_batchnorm(model)
        ...  # forward + backward
        print(find_contaminated_batchnorms.compare(model, before))
    """
    return {name: int(sub.num_batches_tracked)
            for name, sub in model.named_modules()
            if isinstance(sub, _BatchNorm) and sub.num_batches_tracked is not None}
