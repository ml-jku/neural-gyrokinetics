from ..utils import split_args
import torch

SUPPORTED_DROPOUT_MODE = {"vit", "swin", "vista3d"}


def get_act_layer(name: tuple | str):
    """
    Create an activation layer instance.

    For example, to create activation layers:

    .. code-block:: python

        from monai.networks.layers import get_act_layer

        s_layer = get_act_layer(name="swish")
        p_layer = get_act_layer(name=("prelu", {"num_parameters": 1, "init": 0.25}))

    Args:
        name: an activation type string or a tuple of type string and parameters.
    """
    if name == "":
        return torch.nn.Identity()
    act_name, act_args = split_args(name)
    try:
        act_type = getattr(torch.nn, act_name)
    except AttributeError:
        raise AttributeError(
            f"Activation function {name} does not exist, activation name must be called as class "
            f"implemented in pytorch!"
        )
    return act_type(**act_args)
