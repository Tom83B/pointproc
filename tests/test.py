import unittest
from scipy import stats
import numpy as np
from pointproc.densities import *
from pointproc.intensities import *
from pointproc.processes import RenewalProcess, MixedProcess


class TestIntensities(unittest.TestCase):
    def test_addition(self):
        intensity1 = ConstantIntensity()
        intensity2 = ExponentialDecay()
        summed_intensity = intensity1 + intensity2

        params = (1.5, 2, 5)
        expected_result = 2*np.exp(-10/5) + 1.5
        self.assertAlmostEqual(summed_intensity(10, *params), expected_result)


class TestProcesses(unittest.TestCase):
    def test_homog_poisson_fit(self):
        events = np.loadtxt('data/homogenous_poisson_events.txt', delimiter=',')

        intensity = ConstantIntensity(1.5, [0.1, 10])
        density = PoissonDensity()
        process = RenewalProcess(density, intensity)

        process.fit(events, 1000)

        self.assertTrue(process.intensity_params_[0] > 0.9)
        self.assertTrue(process.intensity_params_[0] < 1.1)

    def test_inhomog_poisson_fit(self):
        events = np.loadtxt('data/inhomogenous_poisson_events.txt', delimiter=',')

        intensity = ExponentialDecay() + ConstantIntensity()
        density = PoissonDensity()
        process = RenewalProcess(density, intensity)

        process.fit(events, 1000)
        true_params = np.array([5, 300, 4])

        for p_fitted, p_true in zip(process.intensity_params_, true_params):
            self.assertTrue(p_fitted > p_true * 0.8)
            self.assertTrue(p_fitted < p_true * 1.2)

    def test_inhomog_gamma_fit(self):
        events = np.loadtxt('data/inhomogenous_poisson_events.txt', delimiter=',')

        intensity = ExponentialDecay() + ConstantIntensity()
        density = GammaDensity()
        process = RenewalProcess(density, intensity)

        process.fit(events, 1000)
        true_params = np.array([1, 5, 300, 4])
        fitted_params = np.array([*process.density_params_, *process.intensity_params_])

        for p_fitted, p_true in zip(fitted_params, true_params):
            self.assertTrue(p_fitted > p_true * 0.8)
            self.assertTrue(p_fitted < p_true * 1.2)

    def test_mixture_intensity(self):
        process1 = RenewalProcess(PoissonDensity(), ConstantIntensity())
        process2 = RenewalProcess(PoissonDensity(), ExponentialDecay())
        mixed_process = MixedProcess(process1, process2)
        intensities_params = [1, 10, 2]
        t = 0.5
        vals = mixed_process._intensity_func(t, *intensities_params)
        ints = mixed_process._intensity_integral(t, *intensities_params)

        self.assertAlmostEqual(vals[0], 1)
        self.assertAlmostEqual(vals[1], 10 * np.exp(-t/2))
        self.assertAlmostEqual(ints[0], 0.5)
        self.assertAlmostEqual(ints[1], 20 * (1-np.exp(-t/2)))

    def test_mixture_density(self):
        process1 = RenewalProcess(PoissonDensity(), ConstantIntensity())
        process2 = RenewalProcess(PoissonDensity(), ExponentialDecay())
        mixed_process = MixedProcess(process1, process2)
        intensities_params = [1, 10, 2]
        t = 0.5
        vals = mixed_process._intensity_func(t, *intensities_params)
        ints = mixed_process._intensity_integral(t, *intensities_params)

        d_mix = mixed_process._density_func(vals, ints, 4)
        d1 = process1._density_func(vals[0], ints[0])
        d2 = process1._density_func(vals[1], ints[1])

        self.assertAlmostEqual(d_mix, 0.8*d1 + 0.2*d2)

    def test_mixture_density_array(self):
        process1 = RenewalProcess(PoissonDensity(), ConstantIntensity())
        process2 = RenewalProcess(PoissonDensity(), ExponentialDecay())
        mixed_process = MixedProcess(process1, process2)
        intensities_params = [1, 10, 2]
        t = np.array([0.5, 1, 1.5])

        vals = mixed_process._intensity_func(t, *intensities_params)
        ints = mixed_process._intensity_integral(t, *intensities_params)

        d_mix_arr = mixed_process._density_func(vals, ints, 4)
        d1_arr = process1._density_func(vals[:,0], ints[:,0])
        d2_arr = process1._density_func(vals[:,1], ints[:,1])


        for d_mix, d1, d2 in zip(d_mix_arr, d1_arr, d2_arr):
            self.assertAlmostEqual(d_mix, 0.8*d1 + 0.2*d2)

    def test_poisson_mixture_fit(self):
        events = np.loadtxt('../tests/data/mixed_process_events.txt')[1:]

        process1 = RenewalProcess(PoissonDensity(), ConstantIntensity(init=10))
        process2 = RenewalProcess(GammaDensity(), ConstantIntensity())
        process = MixedProcess(process1, process2)

        process.fit(events, 1000)

        fitted_params = np.array([*process.density_params_, *process.intensity_params_])
        true_params = np.array([1, 0.7 / 0.3, 50, 1])

        for p_fitted, p_true in zip(fitted_params, true_params):
            self.assertTrue(p_fitted > p_true * 0.9)
            self.assertTrue(p_fitted < p_true * 1.1)

    def test_homog_invgauss_fit(self):
        events = np.loadtxt('data/homogenous_invgauss_events.txt', delimiter=',')[1:]

        intensity = ConstantIntensity()
        density = InvGaussDensity()
        process = RenewalProcess(density, intensity)

        process.fit(events, 1000)
        true_params = np.array([0.3, 10])
        fitted_params = np.array([*process.density_params_, *process.intensity_params_])

        for p_fitted, p_true in zip(fitted_params, true_params):
            self.assertTrue(p_fitted > p_true * 0.9)
            self.assertTrue(p_fitted < p_true * 1.1)

if __name__ == '__main__':
    unittest.main()
