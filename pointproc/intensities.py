import numpy as np
from .utils import join_names


class IntensityFunction:
    def __init__(self, func, integral, init_params, bounds, param_names=None):
        self._func = func
        self._integral = integral
        self.x0 = init_params
        self.bounds = bounds

        if param_names is None:
            self.param_names = tuple('p{i+1}' for i in range(len(init_params)))
        else:
            self.param_names = tuple(param_names)

    def __call__(self, t, *params):
        return self._func(t, *params)

    def integral(self, t, *params):
        return self._integral(t, *params)

    def __add__(self, other):
        n_params = len(self.x0)

        def new_func(t, *params):
            self_params = params[:n_params]
            other_params = params[n_params:]

            return self(t, *self_params) + other(t, *other_params)

        def new_integral(t, *params):
            self_params = params[:n_params]
            other_params = params[n_params:]

            return self.integral(t, *self_params) + other.integral(t, *other_params)

        new_init_params = [*self.x0, *other.x0]
        new_bounds = [*self.bounds, *other.bounds]
        new_param_names = join_names(self.param_names, other.param_names)
        return IntensityFunction(new_func, new_integral, new_init_params, new_bounds, new_param_names)


class ConstantIntensity(IntensityFunction):
    def __init__(self, init=1, bounds=(1e-3, None)):
        def func(t, intensity): return np.ones_like(t) * intensity
        def integral(t, intensity): return t * intensity

        param_names = ('const',)

        super(ConstantIntensity, self).__init__(func, integral, [init], [bounds], param_names)


class ExponentialDecay(IntensityFunction):
    def __init__(self, init=None, bounds=None):
        if init is None:
            init = [1, 100]

        if bounds is None:
            bounds = [(0, None), (1e-3, None)]

        param_names = ('ampl', 'tau')

        def func(t, A, tau): return A * np.exp(-t / tau)
        def integral(t, A, tau): return A * tau * (1 - np.exp(-t / tau))

        super(ExponentialDecay, self).__init__(func, integral, init, bounds, param_names)


class Heaviside(IntensityFunction):
    def __init__(self, init=None, bounds=None, sign='positive'):
        self.sign = sign

        if init is None:
            init = 0
        if bounds is None:
            bounds = (0, None)

        if sign == 'positive':
            param_names = ('start',)

            def func(t, x0):
                return np.heaviside(t-x0, 1)

            def integral(t, x0):
                return t - np.clip(t, None, x0)

        elif sign == 'negative':
            param_names = ('end',)

            def func(t, x0):
                return 1-np.heaviside(t-x0, 1)

            def integral(t, x0):
                return np.clip(t, None, x0)

        else:
            raise ValueError('argument sign has to be either "positive" or "negative"')

        super(Heaviside, self).__init__(func, integral, [init], [bounds], param_names)

    def __mul__(self, other):
        n_params = len(self.x0)
        assert n_params == 1

        def new_func(t, *params):
            self_params = params[:n_params]
            other_params = params[n_params:]
            x0 = self_params[0]
            if self.sign == 'positive':
                return self(t, *self_params) * other(t-x0, *other_params)
            else:
                return self(t, *self_params) * other(t, *other_params)

        def new_integral(t, *params):
            self_params = params[:n_params]
            other_params = params[n_params:]
            x0 = self_params[0]
            border_val = other.integral(x0, *other_params)
            if self.sign == 'positive':
                return other.integral(t-x0, *other_params) * np.heaviside(t-x0, 1)
            else:
                return other.integral(t, *other_params) * 1-np.heaviside(t-x0, 1)

        new_init_params = [*self.x0, *other.x0]
        new_bounds = [*self.bounds, *other.bounds]
        new_param_names = join_names(self.param_names, other.param_names)
        return IntensityFunction(new_func, new_integral, new_init_params, new_bounds, new_param_names)


# class TriphasicResponse_(IntensityFunction):
#     def __init__(self, phase1, phase3, init=None, bounds=None):
#         inhibition_param_names = ('response_end', 'gap')
#         response_param_names = tuple('response_'+x for x in phase1.param_names)
#         rebound_param_names = tuple('rebound_'+x for x in phase3.param_names)