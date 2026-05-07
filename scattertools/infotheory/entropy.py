import itertools

import numpy
import numpy as np
import os
from pathlib import Path
import pandas
import sys
import shutil

from numpy.random import permutation
from os import path, mkdir
from math import fabs
from IPython.display import clear_output

from scattertools.support import molstat
from scattertools.infotheory import MVN
from scattertools.infotheory import GMM

from pse.gp import Gp
from pse.gp_server import GpServer


# static methods
def average(alist):
    while True:
        s = np.std(alist)
        result = np.mean(alist)
        maxs = 0
        val = 0
        for element in alist:
            s2 = fabs(result - element)
            if s2 > maxs:
                maxs = s2
                val = element
        if maxs > 3 * s:
            alist.remove(val)
        else:
            break
    return result, s


# multidimensional convolution of nearest horizontal, vertical, and diagonal neighbors but not the center
def convolute(a):
    conv_arr = np.zeros(a.shape)
    offsetshape = tuple([3] * len(a.shape))
    offsetparent = np.zeros(offsetshape)
    it = np.nditer(conv_arr, flags=['multi_index'])
    while not it.finished:
        itindex = tuple(it.multi_index[i] for i in range(len(a.shape)))
        it2 = np.nditer(offsetparent, flags=['multi_index'])
        result = 0.0
        counter = 0.0
        while not it2.finished:
            itindex2 = tuple(it2.multi_index[i] - 1 for i in range(len(a.shape)))
            allzero = True
            for element in itindex2:
                if element != 0:
                    allzero = False
            if not allzero:
                index = tuple(itindex[i] + itindex2[i] for i in range(len(itindex)))
                inarray = True
                for i, element in enumerate(index):
                    if element < 0 or element >= a.shape[i]:
                        inarray = False
                if inarray:
                    result += a[index]
                    counter += 1.0
            it2.iternext()

        conv_arr[itindex] = result / counter
        it.iternext()

    return conv_arr


def rm_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def running_mean(current_mean, n, new_point):
    # see Tony Finch, Incremental calculation of weighted mean and variance
    return current_mean * (n - 1) / n + new_point * (1 / n)


def running_sqstd(current_sqstd, n, new_point, previous_mean, current_mean):
    # see Tony Finch, Incremental calculation of weighted mean and variance
    return (current_sqstd * (n - 1) + (new_point - previous_mean) * (new_point - current_mean)) / n

# Derive the server class using the Entropy(Gp) object below
class Entropy_server(GpServer):
    def __init__(self, **kwargs):
        super().__init__()

    def pse_go(self, data, from_pause=False):
        # TODO: Remove this test after a full elimination of the client argument has been implemented via consistent
        #  subclassing of GP_server
        if 'client' in data:
            del data['client']
        return self.start_Gp_thread(data, from_pause=from_pause, gpobject=Entropy)

# calculates entropy while varying a set of parameters in parlist and keeping others fixed as specified in simpar.dat
# requires a compiled and ready to go fit whose fit parameters are modified and fixed
# avoid_symmetric prevents calculating symmetry-related results by enforcing the indices of varied parameters
# are an ordered list
class Entropy(Gp):
    def __init__(self,
                 exp_par,
                 fitsource,
                 spath,
                 mcmcpath,
                 runfile: str,
                 mcmcburn=16000,
                 mcmcsteps=5000,
                 deldir=True,
                 convergence=2.0,
                 fitter='MCMC',
                 remove_fit_dir=True,
                 lm_iterations=3,
                 mode='water',
                 background_rule=None,
                 configuration=None,
                 qmin=None,
                 qmax=None,
                 qrangefromfile=False,
                 t_total=None,
                 calc_symmetric=True,
                 upper_info_plotlevel=None,
                 plotlimits_filename='',
                 jupyter_clear_output=False,
                 storage_path = None,
                 acq_func="variance",
                 gpcam_iterations=50,
                 gpcam_init_dataset_size=20,
                 gpcam_step=1,
                 keep_plots=False,
                 miniter=1,
                 optimizer='grid',
                 parallel_measurements=1,
                 resume=True,
                 signal_estimate=10,
                 show_support_points=False,
                 train_global_every=None,
                 gp_discrete_points=None,
                 project_name=''):

        """
        The object for an experimental optimization of a SANS or reflectometry experiment, derived from a PSE
        optimization parent object.

        Several Comments:
        * Entropypar.dat contains a list of all fit parameters with a designation, whether they are marginal (d) or
        nuisance (i) parameters. This is followed by the parameter name, the initial parameter value, and the fit
        boundaries.
        * Configuration parameters are given by a preceeding nxy, where x is the data set it applies to and y the
        configuration number. Not giving any xy makes this parameter apply to all configurations, as does passing
        a '*' in place of either x or y.
        * Any number xy following a fit parameter indicates that this paramter is used for this particular
        dataset/configuration to determine the background (incoherent cross-section). This typically applies to SLDs.
        The 'mode' argument for entropy.Entropy() then determines whether this is to be interpreted as an aqueous
        solvent or other.
        * If three more numbers are given, this designates that an information content search over this parameter is
        performed (start, stop, step).
        * A preceding f (fi or fd) at the beginning of the line indicates that the fit boundaries for such a search
        parameter are fixed (for example for volume fractions between 0 and 1). Otherwise, the fit boundary moves
        according to the varied parameter and the initally given fit boundaries.
        * Any theta offset currently needs to have an initial value of zero. Otherwise, refl1d will shift the q-values
        during data simulation with unexpected outcomes.
        * If an instrumental parameter is specified for one data set, the instrumental parameter needs to be specified
        for all other datasets, as well.

        Here is an example file content:
        text = ['d _ _ radius_equatorial 11 5 17',
                'd _ _ radius_polar  20 17 35',
                'i _ _ volfraction  0.01 0.0001 0.02',
                'n * * lambda 6.21',
                'n * * differential_cross_section_buffer 0.059',
                'n * 0 sample_detector_distance 100',
                'n * 1 sample_detector_distance 400',
                'n * 2 sample_detector_distance 1300',
                'n * 0 source_sample_distance 387.6',
                'n * 1 source_sample_distance 850.05',
                'n * 2 source_sample_distance 1467',
                'n * 0 neutron_flux 9e5',
                'n * 1 neutron_flux 2e5',
                'n * 2 neutron_flux 1e5',
                'n * * source_aperture_radius 2.54',
                'n * * sample_aperture_radius 0.635',
                'n * * dlambda_lambda 0.136',
                'n * * beamstop_diameter 10.16',
                'n * 0 time 1600',
                'n * 1 time 3600',
                'n * 2 time 4400',
                'n * 0 beam_center_x 26.416',
                'n * * cuvette_thickness 0.2 0 0 0.02 4.1 0.1'
        ]
        If provided directly as a Pandas dataframe, here are the header names:
        header_names = ['type', 'dataset', 'configuration', 'par', 'value', 'l_fit', 'u_fit', 'l_opt', 'u_opt',
                        'step_opt']

        * Data filenames are currently limited to sim.dat for a single file fit, or simx.dat, x = 0 ... n, for fits
        with multiple data sets

        :param exp_par: (Pandas dataframe | (str | os.PathLike, Path) Contents of an entropy.par file (see above) given
                        either as a Pandas dataframe directly, or as a string / path object pointing to a file from
                        which the information will be read.

        :param fitsource:               CMolStat fitsource
        :param spath:                   CMolStat spath. This is where a prepared fit is provided.
        :param mcmcpath:                CMolStat mcmcpath. A MCMC storage directory within spath for reloading a
                                        problem.
        :param runfile:                 CMolStat runfile

        :param mcmcburn:
        :param mcmcsteps:
        :param deldir:
        :param convergence:
        :param fitter:
        :param remove_fit_dir:
        :param lm_iterations:
        :param mode:
        :param background_rule:
        :param configuration:
        :param qmin:
        :param qmax:
        :param qrangefromfile:
        :param t_total:
        :param calc_symmetric:
        :param upper_info_plotlevel:
        :param plotlimits_filename:
        :param jupyter_clear_output:

        :param storage_path:            PSE Gp storage_path
        :param acq_func:                PSE Gp acq_func
        :param gpcam_iterations:        PSE Gp gpcam_iterations
        :param gpcam_init_dataset_size: PSE Gp gpcam_init_dataset_size
        :param gpcam_step:              PSE Gp gpcam_step
        :param keep_plots:              PSE Gp keep_plots
        :param miniter:                 PSE Gp miniter
        :param optimizer:               PSE Gp optimizer
        :param parallel_measurements:   PSE Gp parallel_measurements
        :param resume:                  PSE Gp resume
        :param signal_estimate:         PSE Gp signal_estimate
        :param show_support_points:     PSE Gp show_support_points
        :param train_global_every:      PSE Gp train_global_every
        :param gp_discrete_points:      PSE Gp gp_discrete_points
        :param project_name:            PSE Gp project_name
        """

        # initialize molstat
        self.fitsource = fitsource
        self.molstat_path = Path(spath).expanduser().resolve()
        self.mcmcpath = mcmcpath
        self.runfile = runfile
        self.molstat = molstat.CMolStat(fitsource=fitsource, spath=spath, mcmcpath=mcmcpath, runfile=runfile)

        # arguments for running the fit
        self.mcmcburn = mcmcburn
        self.mcmcsteps = mcmcsteps
        self.deldir = deldir
        self.convergence = convergence
        self.fitter = fitter
        self.lm_iterations = lm_iterations
        self.remove_fit_dir = remove_fit_dir

        # PSE object arguments
        self.pse_path = storage_path
        self.acq_func = acq_func
        self.gpcam_iterations = gpcam_iterations
        self.gpcam_init_dataset_size = gpcam_init_dataset_size
        self.gpcam_step = gpcam_step
        self.keep_plots = keep_plots
        self.miniter = miniter
        self.optimizer = optimizer
        self.parallel_measurements = parallel_measurements
        self.resume = resume
        self.signal_estimate = signal_estimate
        self.show_support_points = show_support_points
        self.train_global_every = train_global_every
        self.gp_discrete_points = gp_discrete_points
        self.project_name = project_name

        # Data simulation parameters
        self.background_rule = background_rule
        self.configuration = configuration
        self.qmin = qmin
        self.qmax = qmax
        self.qrangefromfile = qrangefromfile
        self.t_total = t_total

        # plotting parameters
        self.upper_info_plotlevel = upper_info_plotlevel
        self.plotlimits_filename = plotlimits_filename
        self.jupyter_clear_output = jupyter_clear_output

        # general optimization parameters
        self.mode = mode
        self.calc_symmetric = calc_symmetric

        # in case we receive the dataframe as a JSONified dict or list
        if isinstance(exp_par, dict) or isinstance(exp_par, list):
            exp_par = pandas.DataFrame(exp_par)

        # Use provided experimental optimization pars or load from file entropypar.dat
        if isinstance(exp_par, pandas.DataFrame):
            # change to canonical names, if necessary
            exp_par = exp_par.rename(columns={'config.': 'configuration', 'parameter': 'par'})
            self.allpar = exp_par
            cols = ["value", "l_fit", "u_fit", "l_opt", "u_opt", "step_opt"]
            for col in cols:
                self.allpar[col] = pandas.to_numeric(self.allpar[col], errors="coerce")
            # TODO: Checks on provided data
        else:
            if exp_par is None:
                filepath = 'entropypar.dat'
            else:
                filepath = Path(exp_par).expanduser().resolve()
            header_names = ['type', 'dataset', 'configuration', 'par', 'value', 'l_fit', 'u_fit', 'l_opt', 'u_opt',
                            'step_opt']
            self.allpar = pandas.read_csv(filepath, sep='\s+', header=None, names=header_names,  skip_blank_lines=True,
                                          comment='#')

        # define unique names, since instrument parameters might have the same name for different datasets and
        # configurations
        self.allpar['unique_name'] = ''
        for i in range(len(self.allpar['par'])):
            if 'n' in self.allpar['type'].iloc[i]:
                datastring = str(self.allpar['dataset'].iloc[i])
                if datastring == '*':
                    datastring = 'x'
                configstring = str(self.allpar['configuration'].iloc[i])
                if configstring == '*':
                    configstring = 'x'
                unique_name = self.allpar['par'].iloc[i] + '_' + datastring + '_' + configstring
            else:
                unique_name = self.allpar['par'].iloc[i]
            self.allpar.iloc[i, self.allpar.columns.get_loc("unique_name")] = unique_name

        # identify dependent (a), independent (b), and non-parameters in simpar.dat for the calculation of p(a|b,y)
        # later on. It is assumed that all parameters in setup.cc are also specified in simpar.dat in exactly the same
        # order. This might have to be looked at in the future.
        # keys: i: independent (nuisance parameter), d: dependent (parameter of interest), n or otherwise none
        # want to calculate H(d|i,y)
        self.dependent_parameters = []
        self.independent_parameters = []
        self.parlist = []

        i = 0
        for row in self.allpar.itertuples():
            if row.type == 'i' or row.type == 'fi':
                self.independent_parameters.append(row.par)
                self.parlist.append(row.par)
                i += 1
            elif row.type == 'd' or row.type == 'fd':
                self.dependent_parameters.append(row.par)
                self.parlist.append(row.par)
                i += 1

        # only those parameters that will be varied
        self.steppar = self.allpar.dropna(axis=0)

        # now, bring steppar in line with PSE Gp's requirements for the search space
        self.pse_par = self.steppar.loc[:, ['unique_name', 'value', 'l_opt', 'u_opt', 'step_opt']].rename(
            columns={
                'unique_name': 'name',
                'l_opt': 'lower_opt',
                'u_opt': 'upper_opt',
                'step_opt': 'step_opt'
            }
        )
        self.pse_par['type'] = 'parameter'
        self.pse_par['optimize'] = True

        # create data frame for simpar.dat needed by the data simulation routines
        # non-parameters such as qrange and prefactor will be included in simpar, but eventually ignored
        # when simulating the scattering, as they will find no counterpart in the model
        self.simpar = self.allpar.loc[:, ['par', 'value', 'dataset', 'configuration', 'unique_name']]

        self.steplist = []
        self.axes = []
        for row in self.steppar.itertuples():
            steps = int((row.u_opt - row.l_opt) / row.step_opt) + 1
            self.steplist.append(steps)
            axis = []
            for i in range(steps):
                axis.append(row.l_opt + i * row.step_opt)
            self.axes.append(axis)

        self.priorentropy, self.priorentropy_marginal = self.calc_prior()

        if self.pse_path is None:
            self.pse_path = self.molstat_path / 'results'
        else:
            self.pse_path = Path(self.pse_path).expanduser().resolve()

        if self.optimizer == 'grid':
            self.results_mvn = np.full(self.steplist, self.priorentropy)
            self.results_gmm = np.full(self.steplist, self.priorentropy)
            self.results_mvn_marginal = np.full(self.steplist, self.priorentropy_marginal)
            self.results_gmm_marginal = np.full(self.steplist, self.priorentropy_marginal)
            self.n_mvn = np.zeros(self.results_mvn.shape)
            self.n_gmm = np.zeros(self.results_gmm.shape)
            self.n_mvn_marginal = np.zeros(self.results_mvn_marginal.shape)
            self.n_gmm_marginal = np.zeros(self.results_gmm_marginal.shape)
            self.sqstd_mvn = np.zeros(self.results_mvn.shape)
            self.sqstd_gmm = np.zeros(self.results_gmm.shape)
            self.sqstd_mvn_marginal = np.zeros(self.results_mvn_marginal.shape)
            self.sqstd_gmm_marginal = np.zeros(self.results_gmm_marginal.shape)
            self.par_median = np.zeros((len(self.parlist),) + self.results_mvn.shape)
            self.par_std = np.zeros((len(self.parlist),) + self.results_mvn.shape)

            if (self.pse_path / 'MVN_entropy.npy').is_file():
                self.load_results_grid(spath)

        elif self.optimizer == 'gpcam' or optimizer == 'gpCAM':
            pass

        # call PSE Gp superclass
        super().__init__(exp_par=self.pse_par,
                         storage_path=self.pse_path,
                         acq_func=self.acq_func,
                         gpcam_iterations=self.gpcam_iterations,
                         gpcam_init_dataset_size=self.gpcam_init_dataset_size,
                         gpcam_step=self.gpcam_step,
                         keep_plots=self.keep_plots,
                         miniter=self.miniter,
                         optimizer=self.optimizer,
                         parallel_measurements=self.parallel_measurements,
                         resume=self.resume,
                         signal_estimate=self.signal_estimate,
                         show_support_points=self.show_support_points,
                         train_global_every=self.train_global_every,
                         gp_discrete_points=self.gp_discrete_points,
                         project_name=self.project_name
                         )

    def calc_entropy(self, molstat=None, cov=False):

        def _init_pars(parnames_dict_keys):
            independent_pars = []
            dependent_pars = []
            parnames = [key for key in parnames_dict_keys]
            for index in range(len(parnames)):
                if parnames[index] in self.independent_parameters:
                    independent_pars.append(index)
                else:
                    dependent_pars.append(index)
            return independent_pars, dependent_pars, parnames

        if cov:
            # use covariance matrix indicates that LM fit is present with active molstat, problem and state
            parnames_dict_keys = molstat.Interactor.problem.labels()
            independent_pars, dependent_pars, parnames = _init_pars(parnames_dict_keys)

            mvnentropy = MVN.MVNEntropy(cov=molstat.Interactor.problem.cov())
            gmm_entropy = mvn_entropy = mvnentropy.entropy()
            gmm_entropy_marginal = mvn_entropy_marginal = mvnentropy.marginal_entropy(independent_pars=independent_pars)

            points_median = molstat.Interactor.problem.getp()
            points_std = molstat.Interactor.problem.stderr()[0]
        else:
            # MCMC fit
            if molstat is None:
                molstat = self.molstat
            points, parnames_dict_keys, logp = molstat.Interactor.fnLoadMCMCResults()
            independent_pars, dependent_pars, parnames = _init_pars(parnames_dict_keys)

            N_entropy = 10000  # was 10000
            N_norm = 10000  # was 2500

            # Do statistics over points
            points_median = np.median(points, axis=0)
            points_std = np.std(points, axis=0)

            # Use a random subset to estimate density
            if N_norm >= len(logp):
                norm_points = points
            else:
                idx = permutation(len(points))[:N_norm]
                norm_points = points[idx]

            mvnentropy = MVN.MVNEntropy(x=norm_points)
            mvn_entropy = mvnentropy.entropy()
            mvn_entropy_marginal = mvnentropy.marginal_entropy(independent_pars=independent_pars)

            gmmentropy = GMM.GMMEntropy(norm_points)
            gmm_entropy = gmmentropy.entropy(N_entropy)
            gmmentropymarginal = GMM.GMMEntropy(np.delete(norm_points, independent_pars, 1))
            gmm_entropy_marginal = gmmentropymarginal.entropy(N_entropy)

        '''
        # This is the original implementation of the KDN entropy from Kramer et al.
        # It is not very stable and replaced by the GMM entropy instead.

        # Use a different subset to estimate the scale factor between density
        # and logp.
        if N_entropy >= len(logp):
            entropy_points, eval_logp = points, logp
        else:
            idx = permutation(len(points))[:N_entropy]
            entropy_points, eval_logp = points[idx], logp[idx]

        # Calculate Kramer Normalized Entropy
        gmmrho = gmmentropy.score_samples(entropy_points)
        frac = exp(eval_logp) / exp(gmmrho)
        n_est, n_err = mean(frac), std(frac)
        s_est = log(n_est) - mean(eval_logp)
        # s_err = n_err/n_est
        # print(n_est, n_err, s_est/LN2, s_err/LN2)
        # print(np.median(frac), log(np.median(frac))/LN2, log(n_est)/LN2)
        kdn_entropy = s_est / LN2

        dependent_points = entropy_points[:, dependent_pars]
        kdn_entropy_marginal = (-1) * np.mean(gmmentropymarginal.score_samples(dependent_points)) / LN2
        '''

        # return MVN entropy, GMM entropy, conditional MVN entropy, conditional GMM entropy
        return mvn_entropy, gmm_entropy, mvn_entropy_marginal, gmm_entropy_marginal, points_median, points_std, parnames

    def calc_entropy_for_iteration(self, molstat_iter, itlabel: int, cov=False):
        # calculate entropy, dependent parameters == parameters of interest
        # independent parameters == nuisance parameters
        mvn, gmm, mvn_marginal, gmm_marginal, points_median, points_std, parnames = \
            self.calc_entropy(molstat_iter, cov=cov)

        if mvn_marginal is None or gmm_marginal is None:
            bValidResult = False
        else:
            bValidResult = (self.priorentropy_marginal - gmm_marginal > (-0.5) * len(self.dependent_parameters)) and \
                           (self.priorentropy - gmm > (-0.5) * len(self.parlist))

        # no special treatment for first entry necessary, algorithm catches this
        if self.optimizer == 'grid':
            if bValidResult:
                self.gridsearch_writeout_result(itlabel, mvn, gmm, mvn_marginal, gmm_marginal, points_median,
                                                points_std, parnames)
            # save results for every iteration
            self.save_results_grid(self.molstat_path)

        return gmm_marginal

    # calculates prior entropy
    def calc_prior(self):
        priorentropy = 0
        priorentropy_marginal = 0
        for row in self.allpar.itertuples():  # cycle through all parameters
            if row.type == 'd' or row.type == 'fd' or row.type == 'i' or row.type == 'fi':
                priorentropy += np.log(row.u_fit - row.l_fit) / np.log(2)
                # calculate prior entropy for parameters to be marginalized (dependent parameters)
                if row.type == 'd' or row.type == 'fd':
                    priorentropy_marginal += np.log(row.u_fit - row.l_fit) / np.log(2)
        return priorentropy, priorentropy_marginal

    def do_measurement(self, opt_pars, it_label, entry, q):

        if self.fitter == 'LM':
            me = []
            for _ in range(self.lm_iterations):
                marginal_entropy = self.prepare_fit(position=entry['position'], itlabel=it_label)
                me.append(self.priorentropy_marginal - marginal_entropy)
            value = numpy.mean(me)
            variance = numpy.var(me)
        else:
            marginal_entropy = self.prepare_fit(position=entry['position'], itlabel=it_label)
            value = self.priorentropy_marginal - marginal_entropy
            variance = 0

        # THESE THREE LINES NEED DO BE PRESENT IN EVERY DERIVED METHOD
        entry['value'] = value
        entry['variance'] = variance
        q.put(entry)

        return value, variance

    def gridsearch_writeout_result(self, itlabel: int, avg_mvn, avg_gmm, avg_mvn_marginal, avg_gmm_marginal,
                                   points_median, points_std, parnames):
        # writes out entropy and parameter results into numpy arrays

        # create itindex from itlabel
        itindex = np.unravel_index(itlabel, self.steplist)

        if not self.calc_symmetric:
            # since symmetry-related points in the optimization were not calculated twice, the current
            # result is copied to all permutations without repetition of the parameter index
            # this is convenient for all interchangeable parameters, such as multiple solvent contrasts
            indexlist = []
            permutated = itertools.permutations(itindex)
            for element in permutated:
                if element not in indexlist:
                    indexlist.append(element)
        else:
            # otherwise copy out to single parameter
            indexlist = [itindex]

        for index in indexlist:
            self.n_gmm[index] += 1.0
            self.n_mvn[index] += 1.0
            self.n_gmm_marginal[index] += 1.0
            self.n_mvn_marginal[index] += 1.0
            n = self.n_gmm[index]

            old_mvn = self.results_mvn[index]
            old_gmm = self.results_gmm[index]
            old_mvn_marginal = self.results_mvn_marginal[index]
            old_gmm_marginal = self.results_gmm_marginal[index]

            self.results_mvn[index] = running_mean(self.results_mvn[index], n, avg_mvn)
            self.results_gmm[index] = running_mean(self.results_gmm[index], n, avg_gmm)
            self.results_mvn_marginal[index] = running_mean(self.results_mvn_marginal[index], n, avg_mvn_marginal)
            self.results_gmm_marginal[index] = running_mean(self.results_gmm_marginal[index], n, avg_gmm_marginal)

            for i in range(self.par_median.shape[0]):
                # parameter indices from fitting results and entropy module might be different
                j = parnames.index(self.parlist[i])
                self.par_median[(i,) + index] = running_mean(self.par_median[(i,) + index], n, points_median[j])
                # for par std the average is calculated, not a sqstd of par_median
                self.par_std[(i,) + index] = running_mean(self.par_std[(i,) + index], n, points_std[j])

            self.sqstd_mvn[index] = running_sqstd(self.sqstd_mvn[index], n, avg_mvn, old_mvn,
                                                  self.results_mvn[index])
            self.sqstd_gmm[index] = running_sqstd(self.sqstd_gmm[index], n, avg_gmm, old_gmm,
                                                  self.results_gmm[index])
            self.sqstd_mvn_marginal[index] = running_sqstd(self.sqstd_mvn_marginal[index], n, avg_mvn_marginal,
                                                           old_mvn_marginal, self.results_mvn_marginal[index])
            self.sqstd_gmm_marginal[index] = running_sqstd(self.sqstd_gmm_marginal[index], n, avg_gmm_marginal,
                                                           old_gmm_marginal, self.results_gmm_marginal[index])

    def gp_hardware_intitialzation(self):
        """
        Method to be implemented in each subclass that initializes the measurement hardware.
        No hardware for this subclass.
        :return: (bool) True if successful, False otherwise.
        """
        return True

    def gp_hardware_shutdown(self):
        """
        Method to be implemented in each subclass that shuts down the measurement hardware.
        No hardware for this subclass.
        :return: (bool) True if successful, False otherwise.
        """
        return True

    def load_results_grid(self, dirname):
        path1 = path.join(dirname, 'results')
        if self.fitter != 'LM':
            self.results_gmm = np.load(path.join(path1, 'GMM_entropy.npy'))
            self.results_gmm_marginal = np.load(path.join(path1, 'GMM_entropy_marginal.npy'))
            self.n_gmm = np.load(path.join(path1, 'GMM_n.npy'))
            self.n_gmm_marginal = np.load(path.join(path1, 'GMM_n_marginal.npy'))
            self.sqstd_gmm = np.load(path.join(path1, 'GMM_sqstd.npy'))
            self.sqstd_gmm_marginal = np.load(path.join(path1, 'GMM_sqstd_marginal.npy'))

        if path.isfile(path.join(path1, 'Prediction_gpcam.npy')):
            self.prediction_gpcam = np.load(path.join(path1, 'Prediction_gpcam.npy'))

        self.results_mvn = np.load(path.join(path1, 'MVN_entropy.npy'))
        self.results_mvn_marginal = np.load(path.join(path1, 'MVN_entropy_marginal.npy'))
        self.n_mvn = np.load(path.join(path1, 'MVN_n.npy'))
        self.n_mvn_marginal = np.load(path.join(path1, 'MVN_n_marginal.npy'))
        self.sqstd_mvn = np.load(path.join(path1, 'MVN_sqstd.npy'))
        self.sqstd_mvn_marginal = np.load(path.join(path1, 'MVN_sqstd_marginal.npy'))
        self.par_median = np.load(path.join(path1, 'par_median.npy'))
        self.par_std = np.load(path.join(path1, 'par_std.npy'))

    def run_fit(self, molstat_instance, iteration, dirname, fulldirname):
        '''
        # run MCMC either cluster or local
        if self.bClusterMode:
            # write runscript
            mcmc_iteration = str(iteration)
            mcmc_dir = dirname
            # replaces the placeholders in slurmscript with variables above
            script = self.slurmscript.format(**locals())

            file = open(path.join(fulldirname, 'runscript'), 'w')
            file.writelines(script)
            file.close()

            lCommand = ['sbatch', path.join(fulldirname, 'runscript')]
            Popen(lCommand)
            self.joblist.append(iteration)
        '''

        if self.fitter == 'MCMC':
            molstat_instance.Interactor.fnRunMCMC(burn=self.mcmcburn, steps=self.mcmcsteps, batch=True)
        else:
            # The best-fit is still loaded in problem from during data simulation. Therefore, we do not
            # reload but use this starting point for a LM, hopefully gaining a speed-up.
            molstat_instance.Interactor.fnRunMCMC(fitter='LM', batch=True, reload_problem=False)
        return

    def plot_results(self, mark_maximum=False):

        super().plot_results(mark_maximum=mark_maximum)

        if self.optimizer != 'grid':
            return

        path1 = path.join(self.molstat_path, 'plots')
        if not path.isdir(path1):
            mkdir(path1)

        if self.fitter != 'LM':
            self.plot_array(self.results_gmm, arr_variance=np.sqrt(self.sqstd_gmm), vallabel='Entropy [bits]',
                              filename=path.join(path1, 'GMM_entropy'))
            self.plot_array(self.results_gmm_marginal, arr_variance=np.sqrt(self.sqstd_gmm_marginal),
                              vallabel='Entropy [bits]', filename=path.join(path1, 'GMM_entropy_marginal'))
            self.plot_array(self.priorentropy - self.results_gmm, arr_variance=np.sqrt(self.sqstd_gmm),
                              vallabel='information gain [bits]', filename=path.join(path1, 'GMM_infocontent'), valmin=0,
                              mark_maximum=mark_maximum)
            self.plot_array(self.priorentropy - self.results_gmm, arr_variance=np.sqrt(self.sqstd_gmm),
                              vallabel='information gain [bits]', filename=path.join(path1, 'GMM_infocontent'), valmin=0,
                              mark_maximum=mark_maximum)
            self.plot_array(self.priorentropy_marginal - self.results_gmm_marginal,
                              arr_variance=np.sqrt(self.sqstd_gmm_marginal), vallabel='information gain [bits]',
                              filename=path.join(path1, 'GMM_infocontent_marginal'), valmin=0,
                              valmax=self.upper_info_plotlevel,
                              mark_maximum=mark_maximum)
            self.plot_array(self.n_gmm, vallabel='computations', filename=path.join(path1, 'GMM_n'), valmin=0)
            self.plot_array(self.n_gmm_marginal, vallabel='computations', filename=path.join(path1, 'GMM_n_marginal'),
                              valmin=0)

        self.plot_array(self.results_mvn, arr_variance=np.sqrt(self.sqstd_mvn), vallabel='Entropy [bits]',
                          filename=path.join(path1, 'MVN_entropy'))
        self.plot_array(self.results_mvn_marginal, arr_variance=np.sqrt(self.sqstd_mvn_marginal),
                          vallabel='Entropy [bits]', filename=path.join(path1, 'MVN_entropy_marginal'))
        self.plot_array(self.priorentropy - self.results_mvn, arr_variance=np.sqrt(self.sqstd_mvn),
                          vallabel='information gain [bits]', filename=path.join(path1, 'MVN_infocontent'), valmin=0,
                          mark_maximum=mark_maximum)
        self.plot_array(self.priorentropy_marginal - self.results_mvn_marginal,
                          arr_variance=np.sqrt(self.sqstd_mvn_marginal), vallabel='information gain [bits]',
                          filename=path.join(path1, 'MVN_infocontent_marginal'), valmin=0, valmax=self.upper_info_plotlevel,
                          mark_maximum=mark_maximum)
        self.plot_array(self.n_mvn, vallabel='computations', filename=path.join(path1, 'MVN_n'), valmin=0)
        self.plot_array(self.n_mvn_marginal, vallabel='computations', filename=path.join(path1, 'MVN_n_marginal'),
                          valmin=0)

        for i, parname in enumerate(self.parlist):
            self.plot_array(self.par_median[i], arr_variance=self.par_std[i], vallabel=parname,
                              filename=path.join(path1, 'Par_' + parname + '_median'))
            self.plot_array(self.par_std[i], arr_variance=None, vallabel=parname,
                              filename=path.join(path1, 'Par_' + parname + '_std'))

    def save_results_grid(self, dirname):
        path1 = path.join(dirname, 'results')
        if not path.isdir(path1):
            mkdir(path1)

        if self.fitter != 'LM':
            np.save(path.join(path1, 'GMM_entropy'), self.results_gmm, allow_pickle=False)
            np.save(path.join(path1, 'GMM_entropy_marginal'), self.results_gmm_marginal, allow_pickle=False)
            np.save(path.join(path1, 'GMM_infocontent'), self.priorentropy - self.results_gmm, allow_pickle=False)
            np.save(path.join(path1, 'GMM_infocontent_marginal'),
                    self.priorentropy_marginal - self.results_gmm_marginal, allow_pickle=False)
            np.save(path.join(path1, 'GMM_sqstd'), self.sqstd_gmm, allow_pickle=False)
            np.save(path.join(path1, 'GMM_sqstd_marginal'), self.sqstd_gmm_marginal, allow_pickle=False)
            np.save(path.join(path1, 'GMM_n'), self.n_gmm, allow_pickle=False)
            np.save(path.join(path1, 'GMM_n_marginal'), self.n_gmm_marginal, allow_pickle=False)

        np.save(path.join(path1, 'MVN_entropy'), self.results_mvn, allow_pickle=False)
        np.save(path.join(path1, 'MVN_entropy_marginal'), self.results_mvn_marginal, allow_pickle=False)
        np.save(path.join(path1, 'MVN_infocontent'), self.priorentropy - self.results_mvn, allow_pickle=False)
        np.save(path.join(path1, 'MVN_infocontent_marginal'), self.priorentropy_marginal - self.results_mvn_marginal,
                allow_pickle=False)
        np.save(path.join(path1, 'MVN_sqstd'), self.sqstd_mvn, allow_pickle=False)
        np.save(path.join(path1, 'MVN_sqstd_marginal'), self.sqstd_mvn_marginal, allow_pickle=False)
        np.save(path.join(path1, 'MVN_n'), self.n_mvn, allow_pickle=False)
        np.save(path.join(path1, 'MVN_n_marginal'), self.n_mvn_marginal, allow_pickle=False)
        np.save(path.join(path1, 'par_median'), self.par_median, allow_pickle=False)
        np.save(path.join(path1, 'par_std'), self.par_std, allow_pickle=False)

        # save to txt when not more than two-dimensional array
        if len(self.steplist) <= 2:
            if self.fitter != 'LM':
                np.savetxt(path.join(path1, 'GMM_entropy.txt'), self.results_gmm - 0)
                np.savetxt(path.join(path1, 'GMM_entropy_marginal.txt'), self.results_gmm_marginal - 0)
                np.savetxt(path.join(path1, 'GMM_infocontent.txt'), self.priorentropy - self.results_gmm)
                np.savetxt(path.join(path1, 'GMM_infocontent_marginal.txt'), self.priorentropy_marginal -
                           self.results_gmm_marginal)
                np.savetxt(path.join(path1, 'GMM_sqstd.txt'), self.sqstd_gmm - 0)
                np.savetxt(path.join(path1, 'GMM_sqstd_marginal.txt'), self.sqstd_gmm_marginal - 0)
                np.savetxt(path.join(path1, 'GMM_n.txt'), self.n_gmm - 0)
                np.savetxt(path.join(path1, 'GMM_n_marginal.txt'), self.n_gmm_marginal - 0)

            np.savetxt(path.join(path1, 'MVN_entropy.txt'), self.results_mvn - 0)
            np.savetxt(path.join(path1, 'MVN_entropy_marginal.txt'), self.results_mvn_marginal - 0)
            np.savetxt(path.join(path1, 'MVN_infocontent.txt'), self.priorentropy - self.results_mvn)
            np.savetxt(path.join(path1, 'MVN_infocontent_marginal.txt'), self.priorentropy_marginal -
                       self.results_mvn_marginal)
            np.savetxt(path.join(path1, 'MVN_sqstd.txt'), self.sqstd_mvn - 0)
            np.savetxt(path.join(path1, 'MVN_sqstd_marginal.txt'), self.sqstd_mvn_marginal - 0)
            np.savetxt(path.join(path1, 'MVN_n.txt'), self.n_mvn - 0)
            np.savetxt(path.join(path1, 'MVN_n_marginal.txt'), self.n_mvn_marginal - 0)
            i = 0
            for parname in self.parlist:
                np.savetxt(path.join(path1, 'Par_' + parname + '_median.txt'), self.par_median[i])
                np.savetxt(path.join(path1, 'Par_' + parname + '_std.txt'), self.par_std[i])
                i += 1

        # save three-dimensional array in slices of the first parameter
        if len(self.steplist) == 3 and self.results_gmm.shape[0] < 6:
            for sl in range(self.results_gmm.shape[0]):
                if self.fitter != 'LM':
                    np.savetxt(path.join(path1, 'GMM_entropy_' + str(sl) + '.txt'), self.results_gmm[sl])
                    np.savetxt(path.join(path1, 'GMM_entropy_marginal_' + str(sl) + '.txt'),
                               self.results_gmm_marginal[sl])
                    np.savetxt(path.join(path1, 'GMM_infocontent_' + str(sl) + '.txt'), self.priorentropy -
                               self.results_gmm[sl])
                    np.savetxt(path.join(path1, 'GMM_infocontent_marginal_' + str(sl) + '.txt'),
                               self.priorentropy_marginal - self.results_gmm_marginal[sl])
                    np.savetxt(path.join(path1, 'GMM_sqstd_' + str(sl) + '.txt'), self.sqstd_gmm[sl])
                    np.savetxt(path.join(path1, 'GMM_sqstd_marginal_' + str(sl) + '.txt'),
                               self.sqstd_gmm_marginal[sl])
                    np.savetxt(path.join(path1, 'GMM_n_' + str(sl) + '.txt'), self.n_gmm[sl])
                    np.savetxt(path.join(path1, 'GMM_n_marginal_' + str(sl) + '.txt'), self.n_gmm_marginal[sl])

                np.savetxt(path.join(path1, 'MVN_entropy_' + str(sl) + '.txt'), self.results_mvn[sl])
                np.savetxt(path.join(path1, 'MVN_entropy_marginal_' + str(sl) + '.txt'),
                           self.results_mvn_marginal[slice])
                np.savetxt(path.join(path1, 'MVN_infocontent_' + str(sl) + '.txt'), self.priorentropy -
                           self.results_mvn[sl])
                np.savetxt(path.join(path1, 'MVN_infocontent_marginal_' + str(sl) + '.txt'),
                           self.priorentropy_marginal - self.results_mvn_marginal[sl])
                np.savetxt(path.join(path1, 'MVN_sqstd_' + str(sl) + '.txt'), self.sqstd_mvn[sl])
                np.savetxt(path.join(path1, 'MVN_sqstd_marginal_' + str(sl) + '.txt'),
                           self.sqstd_mvn_marginal[sl])
                np.savetxt(path.join(path1, 'MVN_n_' + str(sl) + '.txt'), self.n_mvn[sl])
                np.savetxt(path.join(path1, 'MVN_n_marginal_' + str(sl) + '.txt'), self.n_mvn_marginal[sl])

    def set_sim_pars_for_iteration(self, position=None):
        def _str2int(st):
            if st == '*':
                i = 0
            else:
                i = int(st)
            return i

        def _fill_config(configurations, parname, parvalue, dataset, configuration):
            if dataset == '*':
                for ds in range(len(configurations)):
                    if configuration == '*':
                        for cf in range(len(configurations[ds])):
                            configurations[ds][cf][parname] = parvalue
                    else:
                        cf = _str2int(configuration)
                        configurations[ds][cf][parname] = parvalue
            else:
                ds = _str2int(dataset)
                if configuration == '*':
                    for cf in range(len(configurations[ds])):
                        configurations[ds][cf][parname] = parvalue
                else:
                    cf = _str2int(configuration)
                    configurations[ds][cf][parname] = parvalue
            return configurations

        def _set_background(configurations, dset, config, value):
            # calculate background value
            cb = 0
            if self.mode == 'SANS_linear':
                cb = self.background_rule['y_intercept'] + self.background_rule['slope'] * value
            if self.mode == 'water':
                # hardcoded water cb = 0.07 for D2O, cb = 1.00 for H2O
                cb = 0.9245 - 0.1348 * value

            # search if there is a designated background parameter that takes the cb instead of a configuration
            bFoundBackground = False
            for row in self.allpar.itertuples():
                if row.dataset == 'b' + str(dset) or row.dataset == 'b' or row.dataset == 'b*':
                    self.simpar.loc[self.simpar['par'] == row.par, 'value'] = cb
                    bFoundBackground = True
                    if cb < row.l_fit or cb > row.u_fit:
                        print('Parameter with background rule outside of fit boundaries!')
                        print('Parameter: ', row.par)
                        print('Value: ', cb)
                        print('Lower boundary: ', row.l_fit)
                        print('Upper boundary: ', row.u_fit)
                        sys.exit(1)
                    return configurations
            if bFoundBackground:
                return configurations

            # change buffer crosssection in configurations
            configurations = _fill_config(configurations, 'differential_cross_section_buffer', cb, dset, config)

            return configurations

        # Configurations are imported externally, if not empty configuration initialization here
        # In any case, missing parameters are set to the default in the API simulation routines
        configurations = self.configuration
        if configurations is None:
            configurations = [[{}]]

        # cycle through all parameters
        isim = 0
        for row in self.allpar.itertuples():
            simvalue = None
            # is it a parameter to iterate over?
            if row.unique_name in self.steppar['unique_name'].tolist():
                lsim = self.steppar.loc[self.steppar['unique_name'] == row.unique_name, 'l_opt'].iloc[0]
                stepsim = self.steppar.loc[self.steppar['unique_name'] == row.unique_name, 'step_opt'].iloc[0]
                value = self.steppar.loc[self.steppar['unique_name'] == row.unique_name, 'value'].iloc[0]
                lfit = self.steppar.loc[self.steppar['unique_name'] == row.unique_name, 'l_fit'].iloc[0]
                ufit = self.steppar.loc[self.steppar['unique_name'] == row.unique_name, 'u_fit'].iloc[0]

                simvalue = position[isim]

                if row.type == 'd' or row.type == 'fd' or row.type == 'i' or row.type == 'fi':
                    if row.type == 'fd' or row.type == 'fi':
                        # fixed fit boundaries, not floating, for such things as volfracs between 0 and 1
                        lowersim = lfit
                        uppersim = ufit
                    else:
                        lowersim = simvalue - (value - lfit)
                        uppersim = simvalue + (ufit - value)

                    if 'b' not in row.dataset:
                        # only change simpar when it will not be filled by a background rule
                        self.simpar.loc[self.simpar['unique_name'] == row.unique_name, 'value'] = simvalue
                        self.molstat.Interactor.fnReplaceParameterLimitsInSetup(row.par, lowersim, uppersim)
                    else:
                        print('Parameter with background rule cannot be varied!')
                        print('Parameter: ', row.unique_name)
                        raise RuntimeError('Parameter with background rule cannot be varied!')
                else:
                    # must be instrument parameter
                    configurations = _fill_config(configurations, row.par, simvalue, row.dataset, row.configuration)
                    for indx in self.simpar.index:
                        if self.simpar['unique_name'][indx] == row.unique_name and \
                                self.simpar['dataset'][indx] == row.dataset and \
                                self.simpar['configuration'][indx] == row.configuration:
                            self.simpar.iloc[indx, self.simpar.columns.get_loc('value')] = simvalue
                            break

                isim += 1
            elif row.type == 'd' or row.type == 'fd' or row.type == 'i' or row.type == 'fi':
                # We also fill parameters with background rule here. They are supposed to have fit boundaries
                # that encompass all possible values, and the particular row.par does not matter.
                self.molstat.Interactor.fnReplaceParameterLimitsInSetup(row.par, row.l_fit, row.u_fit)
            else:
                # must be instrument parameter
                configurations = _fill_config(configurations, row.par, row.value, row.dataset, row.configuration)

            if (row.type != 'n') and (row.dataset != '_') and ('b' not in row.dataset):
                # this is a parameter that will determine one or more isotropic backgrounds
                if simvalue is None:
                    configurations = _set_background(configurations, row.dataset, row.configuration, row.value)
                else:
                    configurations = _set_background(configurations, row.dataset, row.configuration, simvalue)

        simparsave = self.simpar.loc[:, ['par', 'value']]
        simparsave.to_csv(self.molstat_path / 'simpar.dat', sep=' ', header=None, index=False)
        return configurations


    def prepare_fit(self, position, itlabel: int):

        dirname = 'iteration_' + str(itlabel)
        fulldirname = self.molstat_path / dirname
        path1 = fulldirname / 'save'
        chainname = path1 / (self.runfile+'-chain.mc')

        # most relevant result for a particular index to return for general use of this function
        avg_gmm_marginal = 0

        # LM sometimes produces a singular matrix, which we try to avoid
        fit_counter = 0
        fit_success = False
        while not fit_success:
            # run a new fit, preparations are done in the root directory and the new fit is copied into the
            # iteration directory, preparations in the iterations directory are not possible, because it would
            # be lacking a result directory, which is needed for restoring a state/parameters
            self.molstat.Interactor.fnBackup(target=self.molstat_path / 'simbackup')
            configurations = self.set_sim_pars_for_iteration(position)
            self.molstat.fnSimulateData(mode=self.mode, liConfigurations=configurations, qmin=self.qmin,
                                        qmax=self.qmax, qrangefromfile=self.qrangefromfile, t_total=self.t_total)
            self.molstat.Interactor.fnBackup(origin=self.molstat_path, target=fulldirname)
            # previous save needs to be removed as output serves as flag for HPC job termination
            if path.isdir(path1):
                shutil.rmtree(path1)
            self.molstat.Interactor.fnRemoveBackup(target=path.join(self.molstat_path, 'simbackup'))

            # changing the working directory became necessary at some point for loading the correct data
            os.chdir(fulldirname)
            molstat_iter = molstat.CMolStat(fitsource=self.fitsource, spath=fulldirname, mcmcpath='save',
                                            runfile=self.runfile, load_state=False)

            if self.fitter == 'LM':
                # copy best-fit parameters from data simulation instance
                molstat_iter.Interactor.problem.setp(self.molstat.Interactor.problem.getp())
                self.run_fit(molstat_iter, itlabel, dirname, fulldirname)
                # use covariance matrix for entropy calculation in case of LM
                avg_gmm_marginal = self.calc_entropy_for_iteration(molstat_iter, itlabel=itlabel, cov=True)
                fit_counter += 1
                if avg_gmm_marginal is not None:
                    fit_success = True
                else:
                    if fit_counter > 5:
                        print("Singular matrix encountered for five times in LM.")
                        print("Assume information gain of zero.")
                        avg_gmm_marginal = 0
                        fit_success = True
            else:
                self.run_fit(molstat_iter, itlabel, dirname, fulldirname)
                fit_counter += 1
                fit_success = True

            os.chdir(self.molstat_path)

        # Do not run entropy calculation when no valid result, or entropy from covariance via LM.
        if self.fitter != 'LM':
            bPriorResultExists = path.isfile(str(chainname)) or path.isfile(str(chainname) + '.gz')
            if  bPriorResultExists:
                molstat_iter = molstat.CMolStat(fitsource=self.fitsource, spath=fulldirname, mcmcpath='save',
                                                runfile=self.runfile)
                avg_gmm_marginal = self.calc_entropy_for_iteration(molstat_iter, itlabel=itlabel)

        # delete big files except in Cluster mode. They are needed there for future fetching
        if self.remove_fit_dir:
            shutil.rmtree(fulldirname)
        elif self.deldir :
            rm_file(path.join(path1, self.runfile+'-point.mc'))
            rm_file(path.join(path1, self.runfile+'-chain.mc'))
            rm_file(path.join(path1, self.runfile+'-stats.mc'))
            rm_file(path.join(path1, self.runfile+'-point.mc.gz'))
            rm_file(path.join(path1, self.runfile+'-chain.mc.gz'))
            rm_file(path.join(path1, self.runfile+'-stats.mc.gz'))

        if self.jupyter_clear_output:
            clear_output(wait=True)

        return avg_gmm_marginal

