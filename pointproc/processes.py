from scipy.optimize import minimize, root_scalar
from scipy.stats import expon
import numpy as np
from pointproc.densities import *
from pointproc.intensities import *
from pointproc.utils import concatenate
from tqdm import tqdm
from pointproc.utils import join_names
from scipy.stats import invgauss


class RenewalProcess:
    def __init__(self, density, intensity, deadtime=0, name=None):
        self.density = density
        self.intensity = intensity
        self._density_params = None
        self._intensity_params = None
        self._fitted = False
        self._dnp = len(density.x0)  # number of density parameters
        self._density_init = density.x0
        self._intensity_init = intensity.x0
        self._density_bounds = density.bounds
        self._intensity_bounds = intensity.bounds
        self.deadtime = deadtime

        if name is None:
            self._name = 'p'
        else:
            self._name = name

    def set_params(self, density_params, intensity_params):
        self._density_params = density_params
        self._intensity_params = intensity_params
        self._fitted = True

    def _check_fit(self):
        if not self._fitted:
            raise RuntimeError('Process not fitted')

    def _density_func(self, *args, **kwargs):
        return self.density(*args, **kwargs)

    def _density_integral(self, *args, **kwargs):
        return self.density.integral(*args, **kwargs)

    def _intensity_func(self, *args, **kwargs):
        return self.intensity(*args, **kwargs)

    def _intensity_integral(self, *args, **kwargs):
        return self.intensity.integral(*args, **kwargs)

    @property
    def density_params_(self):
        self._check_fit()
        return self._density_params

    @property
    def intensity_params_(self):
        self._check_fit()
        return self._intensity_params

    @property
    def params_dict_(self):
        self._check_fit()
        dct = {'density': {}, 'intensity': {}}

        for name, val in zip(self.density.param_names, self._density_params):
            dct['density'][name] = val

        for name, val in zip(self.intensity.param_names, self._intensity_params):
            dct['intensity'][name] = val

        return dct

    def _update_init_params(self):
        self._density_init = self._density_params
        self._intensity_init = self._intensity_init

    def _loglikelihood(self, events, tot_time, *params):
        density_params = params[:self._dnp]
        intensity_params = params[self._dnp:]

        events_extended = np.concatenate([[0], events, [tot_time]])

        if self.deadtime == 0:
            integrated_intensity = np.diff(self._intensity_integral(events_extended, *intensity_params), axis=0)
        else:
            interval_starts = events_extended[:-1] + self.deadtime
            interval_ends = events_extended[1:]
            interval_starts[-1] = min(interval_starts[-1], interval_ends[-1])

            integral_starts = self._intensity_integral(interval_starts, *intensity_params)
            integral_ends = self._intensity_integral(interval_ends, *intensity_params)

            integrated_intensity = integral_ends - integral_starts

        intensity = self._intensity_func(events, *intensity_params)
        density = self._density_func(intensity, integrated_intensity[:-1], *density_params)

        n_events_factor = np.log(1 - self._density_integral(integrated_intensity[-1], *density_params))

        logl = np.log(density).sum() + n_events_factor
        # print(logl, params)
        return logl

    def fit(self, events, tot_time):
        def func(x): return -self._loglikelihood(events, tot_time, *x)
        x0 = np.array([*self._density_init, *self._intensity_init])
        bounds = [*self._density_bounds, *self._intensity_bounds]

        min_res = minimize(func, x0=x0, bounds=bounds, method='L-BFGS-B')
        if min_res.success:
            self._density_params = min_res.x[:self._dnp]
            self._intensity_params = min_res.x[self._dnp:]
            self._update_init_params()
            self._fitted = True
        else:
            x = 5
            raise RuntimeError('Optimization did not terminate successfully.')

    def loglikelihood(self, events, tot_time):
        return self._loglikelihood(events, tot_time, *self.density_params_, *self._intensity_params)

    def rescale(self, events):
        self._check_fit()
        if self.deadtime == 0:
            integrated_intensity = np.diff(self._intensity_integral(events, *self._intensity_params), axis=0)
        else:
            interval_starts = events[:-1] + self.deadtime
            interval_ends = events[1:]

            integral_starts = self._intensity_integral(interval_starts, *self._intensity_params)
            integral_ends = self._intensity_integral(interval_ends, *self._intensity_params)

            integrated_intensity = integral_ends - integral_starts
        # integrated_intensity = np.diff(self._intensity_integral(events, *self.intensity_params_), axis=0)
        rescaled_intervals = -np.log(1 - self._density_integral(integrated_intensity, *self.density_params_))
        return rescaled_intervals

    def _int_density_t(self, t, int_intensity0):
        int_intensity1 = self._intensity_integral(t, *self._intensity_params)
        int_intensity_diff = int_intensity1 - int_intensity0
        return self._density_integral(int_intensity_diff, *self._density_params)

    def generate_single_event(self, prev_event):
        self._check_fit()
        int_intensity0 = self._intensity_integral(prev_event, *self._intensity_params) + self.deadtime

        int_intensity_diff = self.density.rvs(*self._density_params)

        def func(t): return self._intensity_integral(t, *self._intensity_params) - (int_intensity0 + int_intensity_diff)

        res = root_scalar(func, x0=prev_event, x1=prev_event+1)

        if res.flag == 'converged':
            return res.root
        else:
            raise RuntimeError('Failed to find root')

    def generate_events(self, tott):
        events = []

        dt = tott / 100
        progress_bar = tqdm(total=100)
        t = 0
        while t < tott:
            t = self.generate_single_event(t)
            events.append(t)
            progress_bar.n = int(t / dt)
            progress_bar.refresh()

        return np.array(events)[:-1]

    def qq_plot(self, events, ax, scatter_params=None, line_params=None):
        self._check_fit()
        rescaled_intervals = self.rescale(events)
        percentiles = np.linspace(0, 100, len(rescaled_intervals))[1:-1]
        theoreticalp = expon.ppf(percentiles / 100)

        if scatter_params is None:
            scatter_params = {}
        if line_params is None:
            line_params = dict(
                c='black',
                linestyle='dashed'
            )

        ax.scatter(np.sort(rescaled_intervals)[1:-1], theoreticalp, **scatter_params)

        xx = np.linspace(rescaled_intervals.min(), rescaled_intervals.max(), 100)
        ax.plot(xx, xx, **line_params)
        ax.set_xlabel('empirical quantiles')
        ax.set_ylabel('theoretical quantiles')


class MixedProcess(RenewalProcess):
    def __init__(self, *processes, deadtime=0):
        self._processes = processes
        self.densities = [proc.density for proc in processes]
        self.intensities = [proc.intensity for proc in processes]
        self._density_params = None
        self._intensity_params = None
        self._weights = None
        self._fitted = False
        self._n_weights = len(processes) - 1
        self._density_init = concatenate([proc.density.x0 for proc in processes])\
                             + [1.] * self._n_weights
        self._intensity_init = concatenate([proc.intensity.x0 for proc in processes])
        self._density_bounds = concatenate([proc.density.bounds for proc in processes])\
                               + [(0, None)] * self._n_weights
        self._intensity_bounds = concatenate([proc.intensity.bounds for proc in processes])
        self._dnp = len(self._density_init)  # number of density parameters
        self._param_lens_d = [len(proc.density.x0) for proc in processes]
        self._param_lens_i = [len(proc.intensity.x0) for proc in processes]
        self._param_sep_i = np.cumsum(self._param_lens_i[:-1])
        self._param_sep_d = np.cumsum(self._param_lens_d)
        self.deadtime = deadtime
        self._names = join_names(*[[p._name] for p in processes])

    def _split_density_params(self, params):
        *param_sets, weights = np.split(params, self._param_sep_d)
        ratio_arr = np.array([*weights, 1])
        ratio_arr = ratio_arr / ratio_arr.sum()
        return param_sets, ratio_arr

    @property
    def params_dict_(self):
        self._check_fit()

        d_param_sets, ratio_arr = self._split_density_params(self._density_params)
        i_param_sets = np.split(self._intensity_params, self._param_sep_i)

        dct = {'weights': ratio_arr}

        for pname, process, dps, ips, w in zip(self._names, self._processes, d_param_sets, i_param_sets, ratio_arr):
            dct[pname] = {'density': {}, 'intensity': {}}

            for name, val in zip(process.density.param_names, dps):
                dct[pname]['density'][name] = val

            for name, val in zip(process.intensity.param_names, ips):
                dct[pname]['intensity'][name] = val

        return dct

    def _density_func(self, intensity, int_intensity_diff, *params):
        param_sets, ratio_arr = self._split_density_params(params)

        return sum([q * proc.density(i, int_i, *ps)
                    for q, i, int_i, ps, proc in zip(ratio_arr, intensity.T, int_intensity_diff.T,
                                                 param_sets, self._processes)])

    def _density_integral(self, int_intensity_diff, *params):
        param_sets, ratio_arr = self._split_density_params(params)

        return sum([q * proc.density.integral(int_i, *ps)
                    for q, int_i, ps, proc in zip(ratio_arr, int_intensity_diff.T,
                                                 param_sets, self._processes)])

    def _intensity_func(self, t, *params):
        param_sets = np.split(params, self._param_sep_i)
        return np.concatenate([[proc.intensity(t, *ps)]
                               for proc, ps in zip(self._processes, param_sets)]).T

    def _intensity_integral(self, t, *params):
        param_sets = np.split(params, self._param_sep_i)
        return np.concatenate([[proc.intensity.integral(t, *ps)]
                               for proc, ps in zip(self._processes, param_sets)]).T

    def fit(self, events, tot_time):
        super(MixedProcess, self).fit(events, tot_time)

        d_param_sets, ratio_arr = self._split_density_params(self._density_params)
        i_param_sets = np.split(self._intensity_params, self._param_sep_i)
        self._weights = ratio_arr

        for i, (dps, ips) in enumerate(zip(d_param_sets, i_param_sets)):
            self._processes[i].set_params(dps, ips)

    def generate_single_event(self, prev_event):
        self._check_fit()
        u = np.random.rand()
        pix = np.digitize(u, np.cumsum(self._weights))
        proc = self._processes[pix]
        return proc.generate_single_event(prev_event)

    def expected_isis(self, times):
        intensities = self._intensity_func(times, *self._intensity_params)
        *param_sets, weights = np.split(self._density_params, self._param_sep_d)

        averages = []

        for i, ps, proc in zip(intensities.T, param_sets, self._processes):
            averages.append(proc.density.average(i, *ps) + self.deadtime)

        return np.array(averages)


class HomogenousProcess(RenewalProcess):
    def __init__(self, density, deadtime=0, name=None):
        intensity = ConstantIntensity()
        super(HomogenousProcess, self).__init__(density, intensity, deadtime, name)

    def generate_single_event(self, prev_event):
        return self.density.rvs(*self._density_params) / self._intensity_params[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from scipy.stats import gamma

    process = RenewalProcess(GammaDensity(init=1.5), ConstantIntensity(init=1)+ExponentialDecay(init=[5,100]))
    process.set_params([1.5], [1, 5, 100])
    events = process.generate_events(300)
    process.fit(events, 300)

    process.params_dict_

    # true_params = [1.5, 10]
    # fitted_params = [process._density_params[0], process._intensity_params[0]]
    #
    # print(true_params)
    # print(fitted_params)