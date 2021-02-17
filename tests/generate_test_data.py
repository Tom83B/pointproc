import numpy as np
from scipy import stats
import warnings


def generate_homogenous_poisson(intensity=1, tott=1000, deadtime=0):
    events = []
    t = 0

    while t < tott:
        events.append(t)
        t += stats.expon.rvs() / intensity + deadtime

    events = np.array(events)

    return events


def generate_inhomog_poisson(intensity_function, tott, dt=1e-3):
    tt = np.arange(0, tott, dt)
    hazard_arr = intensity_function(tt) * dt

    if np.any(hazard_arr > 1e-2):
        warnings.warn('Hazard exceeded 1e-2. Bernoulli approximation might not be appropriate.')

    rand_arr = np.random.rand(len(hazard_arr))
    events_binarized = rand_arr < hazard_arr
    events = np.argwhere(events_binarized).flatten() * dt
    return events


def generate_mixed_process(intensity1, intensity2, q, tott, dt=1e-3):
    events = []
    t = 0

    while t < tott:
        r = np.random.rand()
        if r < q:
            intensity = intensity1
        else:
            intensity = intensity2

        events.append(t)

        t += stats.expon.rvs() / intensity

    events = np.array(events)

    return events


def generate_invgauss_homog(intensity, shape, tott):
    events = []
    t = 0

    while t < tott:
        events.append(t)
        t += stats.invgauss.rvs(mu=shape, scale=1/intensity)

    events = np.array(events)

    return events


if __name__ == '__main__':
    # homogenous_poisson_events = generate_homogenous_poisson(1, 1000)
    # np.savetxt('test_data/homogenous_poisson_events.txt', homogenous_poisson_events, delimiter=',')
    #
    # def intfunc(t):
    #     return 5 * np.exp(-t/300) + 4
    #
    # inhomogenous_poisson_events = generate_inhomog_poisson(intfunc, 1000)
    # np.savetxt('test_data/inhomogenous_poisson_events.txt', inhomogenous_poisson_events, delimiter=',')

    # mixed_process_events = generate_mixed_process(1, 50, 0.3, 1000)
    # np.savetxt('test_data/mixed_process_events.txt', mixed_process_events, delimiter=',')

    # homogenous_invgauss_events = generate_invgauss_homog(10, 0.3, 1000)
    # np.savetxt('test_data/homogenous_invgauss_events.txt', homogenous_invgauss_events, delimiter=',')

    homogenous_poisson_events = generate_homogenous_poisson(10, 1000, 0.2)
    np.savetxt('test_data/homogenous_poisson_events_deadtime.txt', homogenous_poisson_events, delimiter=',')

