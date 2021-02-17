import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import norm, invgauss


class DensityFunction:
    def __init__(self, function, integral, init_params, bounds):
        self._function = function
        self._integral = integral
        self.x0 = init_params
        self.bounds = bounds

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


class GammaDensity(DensityFunction):
    def __init__(self, init=1, bounds=(1e-3, None)):
        def function(intensity, int_intensity_diff, shape):
            x1 = shape * intensity / gamma(shape)
            x2 = (shape * int_intensity_diff) ** (shape - 1)
            x3 = np.exp(-shape * int_intensity_diff)
            return x1 * np.nan_to_num(x2) * x3

        def integral(int_intensity_diff, shape):
            return gammainc(shape, int_intensity_diff)

        super(GammaDensity, self).__init__(function, integral, [init], [bounds])


class InvGaussDensity(DensityFunction):
    def __init__(self, init=1, bounds=(1e-1, None)):
        def function(intensity, int_intensity_diff, shape):
            return intensity * invgauss.pdf(x=int_intensity_diff, mu=shape)

        def integral(int_intensity_diff, shape):
            return invgauss.cdf(mu=shape, x=int_intensity_diff)

        super(InvGaussDensity, self).__init__(function, integral, [init], [bounds])


if __name__ == '__main__':
    density = PoissonDensity()