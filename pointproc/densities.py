import numpy as np
from scipy.stats import gamma, invgauss, expon


class DensityFunction:
    def __init__(self, function, integral, init_params, bounds, param_names=None):
        self._function = function
        self._integral = integral
        self.x0 = init_params
        self.bounds = bounds

        if param_names is None:
            self.param_names = tuple('p{i+1}' for i in range(len(init_params)))
        else:
            self.param_names = tuple(param_names)

    def __call__(self, intensity, int_intensity_diff, *params):
        return self._function(intensity, int_intensity_diff, *params)

    def integral(self, int_intensity_diff, *params):
        return self._integral(int_intensity_diff, *params)


class PoissonDensity(DensityFunction):
    def __init__(self):
        def function(intensity, int_intensity_diff, loc=0):
            return intensity * np.exp(-int_intensity_diff)

        def integral(int_intensity_diff):
            return 1-np.exp(-int_intensity_diff)

        super(PoissonDensity, self).__init__(function, integral, [], [])

    def average(self, intensity):
        return 1 / intensity

    def inverse_cdf(self, q):
        return -np.log(1-q)

    def rvs(self):
        return expon.rvs()


class GammaDensity(DensityFunction):
    def __init__(self, init=1, bounds=(1e-3, None)):
        def function(intensity, int_intensity_diff, shape):
            return shape*intensity * gamma.pdf(x=shape*int_intensity_diff, a=shape)
            # x1 = shape * intensity / gamma(shape)
            # x2 = (shape * int_intensity_diff) ** (shape - 1)
            # x3 = np.exp(-shape * int_intensity_diff)
            # return x1 * np.nan_to_num(x2) * x3

        def integral(int_intensity_diff, shape):
            return gamma.cdf(a=shape, x=shape*int_intensity_diff)

        param_names = ('shape',)

        super(GammaDensity, self).__init__(function, integral, [init], [bounds], param_names)

    def average(self, intensity, shape):
        return 1 / intensity

    def rvs(self, shape):
        return gamma.rvs(a=shape) / shape


class InvGaussDensity(DensityFunction):
    def __init__(self, init=1, bounds=(1e-2, None)):
        def function(intensity, int_intensity_diff, shape):
            return intensity * invgauss.pdf(x=int_intensity_diff, mu=shape)

        def integral(int_intensity_diff, shape):
            return invgauss.cdf(mu=shape, x=int_intensity_diff)

        param_names = ('shape',)

        super(InvGaussDensity, self).__init__(function, integral, [init], [bounds], param_names)

    def average(self, intensity, shape):
        return shape / intensity

    def inverse_cdf(self, q, shape):
        return invgauss.ppf(mu=shape, q=q)

    def rvs(self, shape):
        return invgauss.rvs(mu=shape)


if __name__ == '__main__':
    density = PoissonDensity()