import torch
from typing import Tuple, Hashable, Collection, Any, Mapping
import enum

import einops

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculates the Damerau–Levenshtein distance between two strings for spelling correction.
    https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance
    """
    if s1 == s2:
        return 0
    string_1_length = len(s1)
    string_2_length = len(s2)
    if not s1:
        return string_2_length
    if not s2:
        return string_1_length
    d = {(i, -1): i + 1 for i in range(-1, string_1_length + 1)}
    for j in range(-1, string_2_length + 1):
        d[(-1, j)] = j + 1

    for i, s1i in enumerate(s1):
        for j, s2j in enumerate(s2):
            cost = 0 if s1i == s2j else 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1, d[(i, j - 1)] + 1, d[(i - 1, j - 1)] + cost  # deletion  # insertion  # substitution
            )
            if i and j and s1i == s2[j - 1] and s1[i - 1] == s2j:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + cost)  # transposition

    return d[string_1_length - 1, string_2_length - 1]

def look_up_option(
    opt_str: Hashable,
    supported: Collection | enum.EnumMeta,
    default: Any = "no_default",
    print_all_options: bool = True,
) -> Any:
    """
    Look up the option in the supported collection and return the matched item.
    Raise a value error possibly with a guess of the closest match.

    Args:
        opt_str: The option string or Enum to look up.
        supported: The collection of supported options, it can be list, tuple, set, dict, or Enum.
        default: If it is given, this method will return `default` when `opt_str` is not found,
            instead of raising a `ValueError`. Otherwise, it defaults to `"no_default"`,
            so that the method may raise a `ValueError`.
        print_all_options: whether to print all available options when `opt_str` is not found. Defaults to True

    Examples:

    .. code-block:: python

        from enum import Enum
        from monai.utils import look_up_option
        class Color(Enum):
            RED = "red"
            BLUE = "blue"
        look_up_option("red", Color)  # <Color.RED: 'red'>
        look_up_option(Color.RED, Color)  # <Color.RED: 'red'>
        look_up_option("read", Color)
        # ValueError: By 'read', did you mean 'red'?
        # 'read' is not a valid option.
        # Available options are {'blue', 'red'}.
        look_up_option("red", {"red", "blue"})  # "red"

    Adapted from https://github.com/NifTK/NiftyNet/blob/v0.6.0/niftynet/utilities/util_common.py#L249
    """
    if not isinstance(opt_str, Hashable):
        raise ValueError(f"Unrecognized option type: {type(opt_str)}:{opt_str}.")
    if isinstance(opt_str, str):
        opt_str = opt_str.strip()
    if isinstance(supported, enum.EnumMeta):
        if isinstance(opt_str, str) and opt_str in {item.value for item in supported}:  # type: ignore
            # such as: "example" in MyEnum
            return supported(opt_str)
        if isinstance(opt_str, enum.Enum) and opt_str in supported:
            # such as: MyEnum.EXAMPLE in MyEnum
            return opt_str
    elif isinstance(supported, Mapping) and opt_str in supported:
        # such as: MyDict[key]
        return supported[opt_str]
    elif isinstance(supported, Collection) and opt_str in supported:
        return opt_str

    if default != "no_default":
        return default

    # find a close match
    set_to_check: set
    if isinstance(supported, enum.EnumMeta):
        set_to_check = {item.value for item in supported}  # type: ignore
    else:
        set_to_check = set(supported) if supported is not None else set()
    if not set_to_check:
        raise ValueError(f"No options available: {supported}.")
    edit_dists = {}
    opt_str = f"{opt_str}"
    for key in set_to_check:
        edit_dist = damerau_levenshtein_distance(f"{key}", opt_str)
        if edit_dist <= 3:
            edit_dists[key] = edit_dist

    supported_msg = f"Available options are {set_to_check}.\n" if print_all_options else ""
    if edit_dists:
        guess_at_spelling = min(edit_dists, key=edit_dists.get)  # type: ignore
        raise ValueError(
            f"By '{opt_str}', did you mean '{guess_at_spelling}'?\n"
            + f"'{opt_str}' is not a valid value.\n"
            + supported_msg
        )
    raise ValueError(f"Unsupported option '{opt_str}', " + supported_msg)

def roll_forward_once(
    model, x, grid, ts, problem_dim: int, bundle_steps: int, predict_delta: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO
    x_p = model(x, grid, ts)
    if x_p.ndim == 4:
        x_p = unbind_time(x_p, d=problem_dim)

    x = unbind_time(x, d=problem_dim)
    window_in = x.shape[1]
    # shift input window forward
    tail_idx = torch.arange(bundle_steps, window_in)
    tgt_idx = torch.arange(window_in - bundle_steps, window_in)
    if predict_delta:
        # add to last bundle if predicting deltas
        x_p = x_p + x[:, tgt_idx, ...]
    x = torch.cat([x[:, tail_idx, ...], x_p], dim=1)
    x = x.flatten(1, 2)

    return x_p, x

def unbind_time(x: torch.Tensor, d: int) -> torch.Tensor:
    return einops.rearrange(x, "bs (t d) h w -> bs t d h w", d=d)

def bind_time(x: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(x, "bs t d h w -> bs (t d) h w")