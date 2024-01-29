import numpy as np
from scipy.integrate import solve_ivp

def simulate(dynamics, t, x, u, dt, exit_event_func=None, verbose=False):
    """ Simulates the dynamics for one timestep. Currently only additive noise is supported.
    Args:
        x (numpy array): The state of the system
        u (numpy array): The controls to be applied
        t (timestep): The timestep to consider for time-varying dynamics.
    Returns:
        x_next (numpy array): The state of the system after applying u for one timestep.
        t_next: the next timestep
        on_switching_surface (bool): if on switching surface, return true
        dx: data point of the vector field. Note that for the second order error,
            the input-output tuple has to be ((0.5(x + x_next), u), dx)
    """
    # TODO: generalize to time-varying dynamics.
    def dynfun(t_, x_):
        return dynamics(t_, x_, u)

    if exit_event_func is not None:
        sol_sim = solve_ivp(dynfun, [t, t+dt], x, events=exit_event_func)
        on_switching_surface = sol_sim.t_events[0].size != 0
    else:
        sol_sim = solve_ivp(dynfun, [t, t+dt], x)
        on_switching_surface = False
    x_next = sol_sim.y[:, -1]
    if x_next.ndim == 2:
        x_next = np.squeeze(x_next)
    # TODO: later if we do crazy control
    # if self.angleIndex is not None:
    #     x_next[self.angleIndex] = angle_normalize(x_next[self.angleIndex])
    t_next = sol_sim.t[-1]
    dx = dynfun(0.5 * (t + t_next), 0.5 * (x + x_next))
    if verbose:
        print("t: %.3f, u: %.3f" % (t_next, u))
        print(x_next)
    return x_next, t_next, on_switching_surface, dx

def rollout_controller(x0, dynamics, controller, t_simulation, dt=0.005, exit_event_func=None):
    xs = []
    us = []
    ts = []
    dxs = []
    x = x0
    t = 0
    xs.append(x)
    ts.append(t)
    while t < t_simulation:
        # Simulate dynamics for one timestep.
        u, extras = controller(t, x)
        us.append(u)
        x, t, on_exit_cond, dx = simulate(dynamics, t, x, u, dt, exit_event_func=exit_event_func)
        # TODO: later
        # If on the switching surface, if so, reset.
        # if on_switching_surface:
        #     x = self.reset_map(x)
        #     reset_count += 1

        # Attach to histories.
        xs.append(x)
        ts.append(t)
        dxs.append(dx)
        if on_exit_cond:
            break
    # add one more control to match the size of xs, us, ts.
    u_final_state, extras = controller(t, x)
    us.append(u_final_state)
    return np.asarray(ts), np.asarray(xs).T, np.asarray(us).T, np.asarray(dxs).T