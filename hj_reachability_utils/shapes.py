""" Utility functions for defining target functions that have simple shapes.
    Author: Marius?
"""
import functools
from typing import Any, Callable, Iterable, List, Mapping, Optional, TypeVar, Union

import numpy as np
import jax
import jax.numpy as jnp

T = TypeVar("T")
Tree = Union[T, Iterable["Tree[T]"], Mapping[Any, "Tree[T]"]]

def multivmap(fun: Callable,
              in_axes: Tree[Optional[np.ndarray]],
              out_axes: Tree[Optional[np.ndarray]] = None) -> Callable:
    """Applies `jax.vmap` over multiple axes (equivalent to multiple nested `jax.vmap`s).

    Args:
        fun: Function to be mapped over additional axes (see `jax.vmap` for more details).
        in_axes: Similar to the specification of `in_axes` for `jax.vmap`, with the main difference being that instead
            of `Optional[int]` for axis specification, it's `Optional[np.ndarray]`. For each corresponding input of
            `fun`, the `np.ndarray` specifies a sequence of axes to `jax.vmap` over; note that these axes are not
            specified directly as a `list` so as not to conflict with the possible structure of `in_axes`. All
            non-`None` leaves of `in_axes` (there must be at least one) must have the same length. This length is the
            number of times `jax.vmap` will be applied to `fun`.
        out_axes: Similar to the specification of `out_axes` for `jax.vmap`, with the main difference being that instead
            of `Optional[int]` for axis specification, it's `Optional[np.ndarray]`. For each corresponding output of
            `fun`, the `np.ndarray` specifies a sequence of additional mapped axes to appear in the output. The length
            of non-`None` leaves of `out_axes` must be the same as the length of non-`None` leaves of `in_axes`; the
            order of both axes specifications corresponds to successive nested `jax.vmap` applications. If not provided,
            `out_axes` defaults to `in_axes`.

    Returns:
        A batched/vectorized version of `fun` with arguments that correspond to those of `fun`, but with (possibly
        multiple per input) extra array axes at positions indicated by `in_axes`, and a return value that corresponds
        to that of `fun`, but with (possibly multiple per output) extra array axes at positions indicated by `out_axes`.

    Raises:
        ValueError: if any specified axes are negative or repeated.
    """

    def get_axis_sequence(axis_array: np.ndarray) -> List:
        axis_list = axis_array.tolist()
        if any(axis < 0 for axis in axis_list):
            raise ValueError(f"All `multivmap` axes must be nonnegative; got {axis_list}.")
        if len(axis_list) != len(set(axis_list)):
            raise ValueError(f"All `multivmap` axes must be distinct; got {axis_list}.")
        for i in range(len(axis_list)):
            for j in range(i + 1, len(axis_list)):
                if axis_list[i] > axis_list[j]:
                    axis_list[i] -= 1
        return axis_list

    multivmap_kwargs = {"in_axes": in_axes, "out_axes": in_axes if out_axes is None else out_axes}
    axis_sequence_structure = jax.tree_util.tree_structure(next(a for a in jax.tree_util.tree_leaves(in_axes) if a is not None).tolist())
    vmap_kwargs = jax.tree_util.tree_transpose(jax.tree_util.tree_structure(multivmap_kwargs), axis_sequence_structure,
                                     jax.tree_map(get_axis_sequence, multivmap_kwargs))
    return functools.reduce(lambda f, kwargs: jax.vmap(f, **kwargs), vmap_kwargs, fun)


def shape_hyperplane(grid, normal, point=None):
    """Creates the implicit surface function (signed distance) for a hyperplane.

    Input Parameters:
    - grid:     Grid structure
    - normal:   list of Column vector specifying the outward normal of the hyperplane.
    - point:    list of Vector specifying a point through which the hyperplane passes.
                Defaults to the origin.

    Output Parameters:
    - data: Output data array (of size grid.shape) containing the implicit surface function.
    """
    # default for point
    if point is None:
        point = [0]*len(grid.shape)

    # Normalize the normal to be a unit vector.
    normal = np.array(normal) / np.linalg.norm(normal)

    # implements the function -n^T @ (x - p) over the entire grid
    data = multivmap(lambda x: jnp.matmul(-normal, (x - np.array(point))),
                           np.arange(grid.ndim))(grid.states)

    not_visible_warning(data)

    return data


def shape_rectangle_by_corners(grid, lower, upper):
    """Creates the implicit surface function for a (hyper)rectangle.

    Creates an implicit surface function (close to signed distance)
    for a coordinate axis aligned (hyper)rectangle specified by its
    lower and upper values along the respective axis.

    Input Parameters:
    - grid:     Grid structure
    - lower:    List specifying the lower axis aligned values (corners) per dimension .
    - upper:    List specifying the upper axis aligned values (corners) per dimension .

    e.g. for a 2D cube with corners (1,1), (1,-1), (-1,-1), (-1,1) we have
    lower = [-1, -1] and upper = [1,1]

    Output Parameters:
    - data: Output data array (of size grid.shape) containing the implicit surface function.
    """
    # Implicit surface function calculation.
    # This is basically the intersection (by max operator) of halfspaces.
    # While each halfspace is generated by a signed distance function,
    # the resulting intersection is not quite a signed distance function.

    data = np.maximum(grid.states[..., 0] - upper[0], lower[0] - grid.states[..., 0])
    for i in range(1, grid.ndim):
        data = np.maximum(data, grid.states[..., i] - upper[i])
        data = np.maximum(data, lower[i] - grid.states[..., i])

    not_visible_warning(data)

    return data


def shape_ellipse(grid, center, radii):
    """ Creates the implicit surface function of an ellipsoid.

    Creates an implicit surface function (actually signed distance except for ellipsoids)
    for a coordinate axis aligned cylinder whose axis runs parallel to the
    coordinate dimensions where radii[i] is 'None'.
    e.g. for a 3D cylinder aligned with the z axis radii=[1,1,None].

    Can be used to create:
    Intervals (in 1D)
    Circles and spheres (when all radii are the same)
    Cylinders (when 1 element in Radii is None)
    Ellipsoid Variants of the above (when the elements in radii are different)

    Input Parameters:
    - grid: a grid object
    - center: center as list/np.array
    - radius: list/np.array of radii in the units of the grid or None

    Output Parameters:
    - data: Output data array (of size grid.shape) containing the implicit surface function.
    """
    # check if center is in right dimensions
    if grid.ndim != len(center):
        raise TypeError("Center point must be of same dimension as the grid.")

    if grid.ndim != len(radii):
        raise TypeError("Radius must be of same dimension as the grid.")

    data = np.zeros(grid.shape)
    # for i in range(grid.states.shape[-1]):
    for i in [i for i, x in enumerate(radii) if x is not None]:
        # if radii is None then ignore this dimension to get a cylinder
        # if radii[i] is not None:
        # import pdb
        # pdb.set_trace()
        data = data + ((grid.states[..., i] - center[i]) / radii[i]) ** 2

    data = np.mean([x for x in radii if x is not None]) * (np.sqrt(data) - 1)
    # Note: The multiplication by mean is sort of arbitrary.
    # We still need to ensure 'even' grids (gradient of 1) with NonDim to get good initial sets.

    not_visible_warning(data)
    return data


def shape_sphere(grid, center, radius):
    """ Creates the implicit surface function of a sphere around center on the grid.

    Input Parameters:
    - grid: a grid object
    - center: center as list/np.array
    - radius: float (Note: this is assumed in the units of the grid)

    Output Parameters:
    - data: Output data array (of size grid.shape) containing the implicit surface function.
    """
    data = shape_ellipse(grid, center, [radius]*grid.states.shape[-1])

    return data


def shape_cylinder(grid, center, radius, ignoreDims):
    """ Creates the implicit surface function of a cylinder around center on the grid.
    Input Parameters:
    - grid: a grid object
    - center: center as list/np.array
    - radius: float (Note: this is assumed in the units of the grid)
    - ignoreDims: list of dimensions to ignore e.g. for a 3D cylinder aligned with the z axis ignoreDims=[2]

    Output Parameters:
    - data: Output data array (of size grid.shape) containing the implicit surface function.
    """
    # create corresponding radii list
    radii = []
    for i in range(grid.ndim):
        if i in ignoreDims:
            radii.append(None)
        else:
            radii.append(radius)

    data = shape_ellipse(grid, center, radii)

    return data


def shape_ellipse_autoscale(grid, center, radii_gridpoints):
    """ Creates the initial value function of an ellipsoid with radii scaled with
    gridpoint distance. radii_gridpoints
    Input Parameters:
    - grid: a grid object
    - center: center as list/np.array
    - radii_gridpoints: list of floats specifying the ellipsoid radii as a multiple of dx in each
    dimension. E.g. for 2D Ellipsoid with 3 and 6 gridpoints respectiveley [3,6]

    Output Parameters:
    - data: Output data array (of size grid.shape) containing the implicit surface function.
    """
    if grid.ndim != len(radii_gridpoints):
        raise TypeError("radii_gridpoints must be of same dimension as the grid.")
    radii = []
    for i, dx in enumerate(grid.spacings):
        radii.append(radii_gridpoints[i] * dx)
    return shape_ellipse(grid, center, np.array(radii))


def not_visible_warning(data):
    if np.all(data > 0) or np.all(data < 0):
        raise RuntimeWarning('Implicit surface not visible because function has single sign on grid')
    return None
