import unittest
from scipy import stats
import numpy as np
from pointproc.densities import *
from pointproc.intensities import *
from pointproc.utils import *
from pointproc.processes import RenewalProcess, MixedProcess, TriphasicResponse


data_folder = 'test_data'


class TestIntensities(unittest.TestCase):
    # use numpy.testing.assert_allclose instead
    def test_addition(self):
        intensity1 = ConstantIntensity()
        intensity2 = ExponentialDecay()
        summed_intensity = intensity1 + intensity2

        params = (1.5, 2, 5)
        expected_result = 2*np.exp(-10/5) + 1.5
        self.assertAlmostEqual(summed_intensity(10, *params), expected_result)

    def test_addition_names(self):
        intensity = ConstantIntensity() + ExponentialDecay() + ExponentialDecay()
        expected_param_names = ('const', 'ampl1', 'tau1', 'ampl2', 'tau2')
        self.assertTupleEqual(expected_param_names, intensity.param_names)


class FitTestCase(unittest.TestCase):
    def assertEqualWithTolerance(self, a, b, min_factor=0.8, max_factor=1.2):
        self.assertTrue(a > b * min_factor)
        self.assertTrue(a < b * max_factor)

    def assertIterablesEqualWithTolerance(self, a, b, min_factor=0.8, max_factor=1.2):
        for x, y in zip(a, b):
            self.assertEqualWithTolerance(x, y, min_factor, max_factor)


class TestRenewalProcess(FitTestCase):
    def test_homog_poisson_fit(self):
        events = np.loadtxt(f'{data_folder}/homogenous_poisson_events.txt', delimiter=',')[1:]

        intensity = ConstantIntensity(1.5, [0.1, 10])
        density = PoissonDensity()
        process = RenewalProcess(density, intensity)

        process.fit(events, 1000)

        self.assertTrue(process.intensity_params_[0] > 0.9)
        self.assertTrue(process.intensity_params_[0] < 1.1)

    def test_inhomog_poisson_fit(self):
        events = np.loadtxt(f'{data_folder}/inhomogenous_poisson_events.txt', delimiter=',')[1:]

        intensity = ExponentialDecay() + ConstantIntensity()
        density = PoissonDensity()
        process = RenewalProcess(density, intensity)

        process.fit(events, 1000)
        true_params = np.array([5, 300, 4])

        for p_fitted, p_true in zip(process.intensity_params_, true_params):
            self.assertTrue(p_fitted > p_true * 0.8)
            self.assertTrue(p_fitted < p_true * 1.2)

    def test_inhomog_gamma_fit(self):
        events = np.loadtxt(f'{data_folder}/inhomogenous_poisson_events.txt', delimiter=',')[1:]

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

    def test_homog_invgauss_fit(self):
        events = np.loadtxt(f'{data_folder}/homogenous_invgauss_events.txt', delimiter=',')[1:]

        intensity = ConstantIntensity(init=15)
        density = InvGaussDensity(init=0.35)
        process = RenewalProcess(density, intensity)

        process.fit(events, 1000)
        true_params = np.array([0.3, 10])
        fitted_params = np.array([*process.density_params_, *process.intensity_params_])

        self.assertIterablesEqualWithTolerance(fitted_params, true_params)

    def test_generate_invgauss(self):
        process = RenewalProcess(InvGaussDensity(init=0.3), ConstantIntensity(init=10))
        process.set_params([0.3], [10])
        events = process.generate_events(10)
        process.fit(events, 10)

        true_params = [0.3, 10]
        fitted_params = [process._density_params[0], process._intensity_params[0]]

        self.assertIterablesEqualWithTolerance(fitted_params, true_params)

    def test_generate_gamma(self):
        process = RenewalProcess(GammaDensity(init=1.5), ConstantIntensity(init=10))
        process.set_params([1.5], [10])
        events = process.generate_events(10)
        process.fit(events, 10)

        true_params = [1.5, 10]
        fitted_params = [process._density_params[0], process._intensity_params[0]]

        self.assertIterablesEqualWithTolerance(fitted_params, true_params)

    def test_generate_inhomog_gamma(self):
        process = RenewalProcess(GammaDensity(init=1.5), ConstantIntensity(init=1) + ExponentialDecay(init=[5, 200]))
        process.set_params([1.5], [1, 5, 200])
        events = process.generate_events(500)
        process.fit(events, 500)

        fitted_params = np.array([*process.density_params_, *process.intensity_params_])
        true_params = [1.5, 1, 5, 200]

        self.assertIterablesEqualWithTolerance(fitted_params, true_params, min_factor=0.7, max_factor=1.3)


class TestMixtureProcess(FitTestCase):
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
        events = np.loadtxt(f'{data_folder}/mixed_process_events.txt')[1:]

        process1 = RenewalProcess(PoissonDensity(), ConstantIntensity(init=10))
        process2 = RenewalProcess(GammaDensity(), ConstantIntensity())
        process = MixedProcess(process1, process2)

        process.fit(events, 1000)

        fitted_params = np.array([*process.density_params_, *process.intensity_params_])
        true_params = np.array([1, 0.7 / 0.3, 50, 1])

        self.assertIterablesEqualWithTolerance(fitted_params, true_params)

    def test_process_probability(self):
        process1 = RenewalProcess(PoissonDensity(), ConstantIntensity(init=10))
        process2 = RenewalProcess(PoissonDensity(), ConstantIntensity())
        process = MixedProcess(process1, process2)

        process.set_params(density_params=[0.7 / 0.3], intensity_params=[50, 1])

        l1, l2 = np.array([50, 1])
        q1, q2 = np.array([0.7, 0.3])

        threshold_theory = - np.log((q2 * l2) / (q1 * l1)) / (l1 - l2)

        isi1, isi2 = threshold_theory / 2, threshold_theory * 1
        events = np.array([0, isi1, isi2]).cumsum()
        isis = np.diff(events)

        dens1 = q1 * l1 * np.exp(-l1 * isis)
        dens2 = q2 * l2 * np.exp(-l2 * isis)
        probs_theory = dens1 / (dens1 + dens2)
        probs_test = process.process_probabilities(events)[0]
        np.testing.assert_almost_equal(probs_theory, probs_test)


class TestTriphasic(FitTestCase):
    def test_poisson_changepoint(self):
        data_folder = '../tests/test_data'

        events = np.loadtxt(f'{data_folder}/triphasic_response_poisson.txt', delimiter=',')[1:]
        process1 = RenewalProcess(PoissonDensity(), ConstantIntensity(init=50))
        process2 = RenewalProcess(PoissonDensity(), ConstantIntensity(init=2))

        tri = TriphasicResponse(process1, process2)
        tri.fit(events, tot_time=1020, resp_end_range=(8, 12))

        last1 = events[events <= 10].max()
        first2 = events[events > 10].min()

        self.assertGreater(tri.sep_time, last1)
        self.assertLess(tri.sep_time, first2)


class TestUtils(unittest.TestCase):
    def test_join_names(self):
        nl1 = ['a', 'b', 'c']
        nl2 = ['d', 'b', 'a']

        joined = join_names(nl1, nl2)
        self.assertListEqual(joined, ['a1', 'b1', 'c', 'd', 'b2', 'a2'])


if __name__ == '__main__':
    unittest.main()
