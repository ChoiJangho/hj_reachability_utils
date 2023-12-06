import abc
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import numpy as np
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator
import hj_reachability as hj

class GridBoundaryType(Enum):
    ExtrapolateAwayFromZero = 0
    Dirichlet = 1
    Periodic = 2

@dataclass
class HjGridMetaData:
    domain_lo: np.ndarray
    domain_hi: np.ndarray
    shape: Tuple[int]
    boundary_types: Tuple[GridBoundaryType]
    dirichlet_value: Optional[float]

@dataclass
class HjData:
    grid_meta_data: HjGridMetaData
    target_function: Optional[np.ndarray]
    constraint_function: Optional[np.ndarray]
    times: np.ndarray
    values: np.ndarray

class ControlAndDisturbanceAffineDynamics(hj.Dynamics):
    """Abstract base class for representing control- and disturbance-affine dynamics."""

    def __call__(self, state, control, disturbance, time):
        """Implements the affine dynamics `dx_dt = f(x, t) + G_u(x, t) @ u + G_d(x, t) @ d`."""
        return (self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control +
                self.disturbance_jacobian(state, time) @ disturbance)

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time):
        """Implements the open loop dynamics `f(x, t)`."""

    @abc.abstractmethod
    def control_jacobian(self, state, time):
        """Implements the control Jacobian `G_u(x, t)`."""

    @abc.abstractmethod
    def disturbance_jacobian(self, state, time):
        """Implements the disturbance Jacobian `G_d(x, t)`."""

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        return (self.control_space.extreme_point(control_direction),
                self.disturbance_space.extreme_point(disturbance_direction))

    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_jacobian(state, time)) @ self.disturbance_space.max_magnitudes)


def get_constant_dirichlet(constant):
    return lambda x, pad_width: jnp.pad(x, ((pad_width, pad_width)), "constant", constant_values=constant)

def get_hj_grid_from_meta_data(meta_data: HjGridMetaData):
    boundary_conditions = []
    for boundary_type in meta_data.boundary_types:
        if boundary_type == GridBoundaryType.Dirichlet:
            dirichlet = get_constant_dirichlet(meta_data.dirichlet_value)
            boundary_conditions.append(dirichlet)
        elif boundary_type == GridBoundaryType.ExtrapolateAwayFromZero:
            boundary_conditions.append(hj.boundary_conditions.extrapolate_away_from_zero)
        elif boundary_type == GridBoundaryType.Periodic:
            boundary_conditions.append(hj.boundary_conditions.periodic)
    boundary_conditions = tuple(boundary_conditions)
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain = hj.sets.Box(lo=meta_data.domain_lo, 
                            hi=meta_data.domain_hi),
        shape = meta_data.shape,
        boundary_conditions=boundary_conditions
    )
    return grid

def get_projected_grid_meta(original_grid: HjGridMetaData, projection_dims: List[int]):
    projected_grid = HjGridMetaData(original_grid.domain_lo[projection_dims],
                                    original_grid.domain_hi[projection_dims],
                                    tuple(original_grid.shape[dim] for dim in projection_dims),
                                    tuple(original_grid.boundary_types[dim] for dim in projection_dims),
                                    original_grid.dirichlet_value)
    return projected_grid

def get_reach_avoid_postprocessor(target_function, constraint_function):
    """
        target_function: negative inside the target set.
        constraint_function: negative inside the target set.
    """
    return lambda t, v: jnp.maximum(jnp.minimum(v, target_function), constraint_function)

def get_avoid_postprocessor(constraint_function):
    """
        constraint_function: negative inside the target set.
    """
    return lambda t, v: jnp.maximum(v, constraint_function)

def compute_reach_avoid_value_function(dynamics: hj.Dynamics,
                                     t_max: float,
                                     num_timestep: int,
                                     grid,
                                     target_function,
                                     constraint_function,
                                     accuracy = "high"):
    assert t_max > 0, "t_max has to be positive."
    times = np.linspace(0, -t_max, num_timestep)
    artificial_dissipation_scheme = hj.artificial_dissipation.global_lax_friedrichs
    reach_avoid_postprocessor = get_reach_avoid_postprocessor(target_function, constraint_function)

    # Create the settings object
    solver_settings = hj.SolverSettings.with_accuracy(
        accuracy=accuracy,
        artificial_dissipation_scheme=artificial_dissipation_scheme,
        value_postprocessor=reach_avoid_postprocessor
    )

    values = hj.solve(
        solver_settings, dynamics, grid, times, target_function
    )

    return times, np.asarray(values)

def compute_avoid_value_function(dynamics: hj.Dynamics,
                                     t_max: float,
                                     num_timestep: int,
                                     grid,
                                     constraint_function,
                                     accuracy = "high"):
    assert t_max > 0, "t_max has to be positive."
    times = np.linspace(0, -t_max, num_timestep)
    artificial_dissipation_scheme = hj.artificial_dissipation.global_lax_friedrichs
    avoid_postprocessor = get_avoid_postprocessor(constraint_function)

    # Create the settings object
    solver_settings = hj.SolverSettings.with_accuracy(
        accuracy=accuracy,
        artificial_dissipation_scheme=artificial_dissipation_scheme,
        value_postprocessor=avoid_postprocessor
    )

    values = hj.solve(
        solver_settings, dynamics, grid, times, constraint_function
    )
    return times, np.asarray(values)

def get_ttr(state,
            times: np.ndarray, 
            grid: hj.Grid, 
            values: np.ndarray):
    """
    find minimal time-to-reach (TTR) by doing binary search.
    """
    len_times = times.shape[0]
    assert values.shape[0] == len_times, "time-varying value function should be given"
    assert times[0] == 0, "initial time should be zero."
    assert times[-1] < 0, "final time should be negative value (indicating backward reachability)."
    def eval_value_at_index(index: int):
        return grid.interpolate(values[index, ...], state)

    def find_first_negative_index():
        low, high = 0, len_times - 1
        # Edge case: Check if the first element is already negative
        if eval_value_at_index(low) < 0:
            return low

        while low <= high:
            mid = (low + high) // 2
            # Check if mid is the first negative element
            if eval_value_at_index(mid) < 0 and (mid == 0 or eval_value_at_index(mid - 1) >= 0):
                return mid
            elif eval_value_at_index(mid) >= 0:
                low = mid + 1
            else:
                high = mid - 1
        # Edge case: the value is not in the BRT of the given time span.
        return len_times - 1

    ttr_index = find_first_negative_index()
    ttr = -times[ttr_index]
    return ttr, ttr_index

def optimal_control_with_time(time, state, 
                              dynamics: hj.Dynamics, 
                              times: np.ndarray, 
                              grid: hj.Grid, 
                              values: np.ndarray):
    ttr, ttr_index = get_ttr(state, times, grid, values)
    print(ttr)
    value = grid.interpolate(values[ttr_index, :, :], state)
    grad_values = grid.grad_values(values[ttr_index, :, :])
    grad_value = grid.interpolate(grad_values, state)
    # print(grad_value)
    u_opt_jnp = dynamics.optimal_control(state, time, grad_value=grad_value)
    extras = {}
    extras['ttr'] = ttr
    extras['value'] = value
    extras['grad_value'] = grad_value
    return np.asarray(u_opt_jnp), extras

def one_step_predictive_control_with_time(time, state, 
                                          dynamics: hj.Dynamics, 
                                          times: np.ndarray, 
                                          grid: hj.Grid, 
                                          values: np.ndarray):
    ttr, ttr_index = get_ttr(state, times, grid, values)
    print(ttr)
    value = grid.interpolate(values[ttr_index, :, :], state)
    u_opt_jnp = dynamics.one_step_predictive_control(state, time, grid, values[ttr_index, :, :])
    extras = {}
    extras['ttr'] = ttr
    extras['value'] = value
    return np.asarray(u_opt_jnp), extras

def numerical_gradient(func, dims, eps=1e-8):
    assert (len(dims) == 1 or len(dims) == 2)
    shift_x = np.eye(dims[0])*eps
    if len(dims) == 1:
        def grad(x, t=None):
            return np.asarray([(np.array(func(x+shift_x[:, k], t)) -
                                np.array(func(x-shift_x[:, k], t))) / (2*eps) for k in range(dims[0])]).T
        return grad
    else:
        def grad_x(x, u, t=None):
            return np.asarray([(np.array(func(x+shift_x[:, k], u, t)) -
                                np.array(func(x-shift_x[:, k], u, t))) / (2*eps) for k in range(dims[0])]).T

        def grad_u(x, u, t=None):
            shift_u = np.eye(dims[1]) * eps
            return np.asarray([(np.array(func(x, u+shift_u[:, k], t)) -
                                np.array(func(x, u-shift_u[:, k], t))) / (2*eps) for k in range(dims[1])]).T

        return grad_x, grad_u
    
def migrate_value_to_new_grid(grid_original: hj.Grid, value_original: np.ndarray, grid_new: hj.Grid):
    ndim = grid_original.ndim
    interploation_input_domain = tuple(grid_original.coordinate_vectors[i] for i in range(ndim))
    if value_original.ndim == ndim:
        interpolate_global = RegularGridInterpolator(interploation_input_domain, value_original, fill_value=np.nan, bounds_error=False)
        interploation_output_domain = tuple(grid_new.states[..., i] for i in range(ndim))
        value_new = interpolate_global(interploation_output_domain)
        return value_new
    if value_original.ndim == ndim+1:
        # value function has one more dimension in time.
        num_timestep = value_original.shape[0]
        value_new = np.zeros((num_timestep,) + grid_new.shape)
        for i in range(num_timestep):
            interpolate_global = RegularGridInterpolator(interploation_input_domain, value_original[i, ...], fill_value=np.nan, bounds_error=False)
            interploation_output_domain = tuple(grid_new.states[..., idim] for idim in range(ndim))
            value_new[i, ...] = interpolate_global(interploation_output_domain)
        return value_new