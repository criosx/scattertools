from __future__ import print_function
from math import sqrt
from os import path
from scipy import stats, special
import matplotlib.pyplot as plt
import numpy
import pandas
import os
import shutil
import bumps.curve


def prepare_fit_directory(fitdir=None, runfile=None, datafile_names=None):
    """
    Makes or empties a fit directory and copies a bumps-style runfile and the data to this directory.
    :param fitdir: path to the fit directory
    :param runfile: bumps-style runfile
    :param datafile_names: list of paths to the data
    :return: None
    """
    if not os.path.isdir(fitdir):
        os.mkdir(fitdir)
    else:
        for f in os.listdir(fitdir):
            fpath = os.path.join(fitdir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
    # copy script and runfiles into fitdir
    shutil.copyfile(runfile, os.path.join(fitdir, os.path.basename(runfile)))
    for file in datafile_names:
        if os.path.isfile(file):
            shutil.copyfile(file, os.path.join(fitdir, os.path.basename(file)))


class CMolStat:
    def __init__(self, fitsource="refl1d", spath=".", mcmcpath=".",
                 runfile="run", state=None, problem=None,
                 load_state=True, save_stat_data=False):
        """
        self.diParameters is a dictionary containing all parameters. Structure:

        dictionary: sparameter : dictionary
                                 'number'    : int       # order of initialization in ga_refl
                                 'lowerlimit'  : float   # constraint range lower limit
                                 'upperlimit'  : float   # constraint range upper limit
                                 'value'  : float        # absolute value
                                 'error'  : float        # error derived from covariance matrix
                                 'relval' : float        # relative value between 0 and 1 in terms of the constraints
                                 'variable': string       # associated ga_refl variable

        self.diStatResults is a dictionary of statistical results with entries from various routines:
            (from fnAnalyzeStatFile)
            'Parameters' is itself a dictionary with the following keys:
                      'Values' key contains ordered list of all MCMC parameters
                      'LowPerc' contains lower percentile
                      'Median' contains median
                      'HighPerc' contains higher percentile
                      'MaxParameterLength' contains length of the longest parameter name

            (from fnLoadStatData)
            'NumberOfStatValues' number of MC points
            'nSLDProfiles' contains all profiles 'nSLDProfiles'[MCiteration][model][z,rho]
            'Molgroups' contains a list of dictionaries storing all molgroups
                'Molgroups'[MCiteration] {'molgroupname' {'zaxis' [z]
                                                          'are' [area]
                                                          'sl' [sl]
                                                          'sld' [sld]
                                                          'headerdata' {'property' value}
                                                          'property1' value
                                                          'property2' value}}
           'Results' contains a dictionary of derived values from fit paramters for post-analysis
                'Results' is itself a dictionary with the following keys:
                        'Values' key contains ordered list of all MCMC derived parameters
        """

        self.diParameters = {}
        self.diMolgroups = {}
        self.diStatResults = {}
        self.diResults = {}

        self.fitsource = fitsource  # define fitting software
        self.spath = spath
        self.mcmcpath = mcmcpath

        # check if runfile has .py ending, if yes, strip it
        if os.path.splitext(runfile)[1] == '.py':
            self.runfile = os.path.splitext(runfile)[0]
        else:
            self.runfile = runfile

        self.liStatResult = []  # a list that contains the isErr.dat or sErr.dat file line by line
        self.sStatResultHeader = ''  # headerline from isErr.dat or sErr.dat
        self.sStatFileName = ''  # Name of the statistical File
        self.chisq = 0.
        self.fMolgroupsStepsize = 0.
        self.iMolgroupsDimension = 0
        self.fMolgroupsNormArea = 0
        # check for system and type of setup file

        self.Interactor = None
        if self.fitsource == "bumps":
            from scattertools.support import api_bumps
            self.Interactor = api_bumps.CBumpsAPI(self.spath, self.mcmcpath, self.runfile, state, problem,
                                                  load_state=load_state)
        elif self.fitsource == 'refl1d':
            from scattertools.support import api_refl1d
            self.Interactor = api_refl1d.CRefl1DAPI(self.spath, self.mcmcpath, self.runfile, load_state=load_state)
        elif self.fitsource == 'garefl':
            from scattertools.support import api_garefl
            self.Interactor = api_garefl.CGaReflAPI(self.spath, self.mcmcpath, self.runfile, load_state=load_state)
        elif self.fitsource == 'SASView':
            from scattertools.support import api_sasview
            self.Interactor = api_sasview.CSASViewAPI(self.spath, self.mcmcpath, self.runfile, load_state=load_state)

        self.save_stat_data = save_stat_data

    def fnAnalyzeStatFile(self, fConfidence=-1, sparse=0):
        """
        Summarizes statistical information from MCMC for parameters, results, and SLD profiles (Molecular groups
        are handled by fnProfilesStat). Results are stored in self.diStatResults['Parameters'] and
        self.diStatResults['Results'].

        :param fConfidence: 0 < confidence < 1, percentile used for statistical analysis
                           confidence < 0: multiple of sigmas used for statistical analysis (-1 -> 1 sigma,
                           -2 -> 2 sigma, ...)
        :param sparse: 0 < sparse < 1, fraction of statistical data from the MCMC that is used the summary
                           sparse > 1, number of iterations from the MCMC used for summary

        :return: Pandas dataframe with statistical results in addition to internal storage of the results
        """

        def data_append(data, origin, name, vis, lower_limit, upper_limit, lower_percentile, median_percentile,
                        upper_percentile, interval_lower, interval_upper, confidence):
            data['origin'].append(origin)
            data['name'].append(name)
            data['vis'].append(vis)
            data['lower limit'].append(lower_limit)
            data['upper limit'].append(upper_limit)
            data['lower percentile'].append(lower_percentile)
            data['median percentile'].append(median_percentile)
            data['upper percentile'].append(upper_percentile)
            data['interval lower'].append(interval_lower)
            data['interval upper'].append(interval_upper)
            data['confidence'].append(confidence)
            return data

        # self.fnLoadParameters()
        self.fnLoadStatData(sparse)

        data = {'origin': [],
                'name': [],
                'vis': [],
                'lower limit': [],
                'upper limit': [],
                'lower percentile': [],
                'median percentile': [],
                'upper percentile': [],
                'interval lower': [],
                'interval upper': [],
                'confidence': []
                }

        confidence = min(fConfidence, 1)
        if confidence < 0:
            confidence = special.erf(-1 * confidence / sqrt(2))
        percentiles = (100.0 * (1 - confidence) / 2, 50., 100.0 - 100.0 * (1 - confidence) / 2)
        iNumberOfMCIterations = self.diStatResults['NumberOfStatValues']

        print('Analysis of MCMC fit ...')
        print('Number of iterations: %(ni)d' % {'ni': iNumberOfMCIterations})
        print('')
        print('Fit Parameters:')

        # Fit Parameters
        for element in sorted(list(self.diParameters.keys()),
                              key=lambda sParameter: self.diParameters[sParameter]['number']):
            vals = self.diStatResults['Parameters'][element]['Values']
            perc = stats.scoreatpercentile(vals, percentiles)
            self.diStatResults['Parameters'][element]['LowPerc'] = perc[0]
            self.diStatResults['Parameters'][element]['Median'] = perc[1]
            self.diStatResults['Parameters'][element]['HighPerc'] = perc[2]

            flowerlimit = self.diParameters[element]['lowerlimit']
            fupperlimit = self.diParameters[element]['upperlimit']
            temp = abs(fupperlimit - flowerlimit)

            sGraphOutput = '['
            itemp1 = int((perc[0] - flowerlimit) / temp * 10 + 0.5)
            itemp2 = int((perc[1] - flowerlimit) / temp * 10 + 0.5)
            itemp3 = int((perc[2] - flowerlimit) / temp * 10 + 0.5)
            for i in range(11):
                s1 = ' '
                if itemp1 == i or itemp3 == i:
                    s1 = '|'
                if itemp2 == i:
                    if s1 == '|':
                        s1 = '+'
                    else:
                        s1 = '-'
                sGraphOutput += s1
            sGraphOutput += ']'
            if (perc[0] - flowerlimit) < temp * 0.01:
                self.diStatResults['Parameters'][element]['LowerLimitCollision'] = True
                sGraphOutput = '#' + sGraphOutput[1:]
            else:
                self.diStatResults['Parameters'][element]['LowerLimitCollision'] = False
            if (fupperlimit - perc[2]) < temp * 0.01:
                self.diStatResults['Parameters'][element]['UpperLimitCollision'] = True
                sGraphOutput = sGraphOutput[:-1] + '#'
            else:
                self.diStatResults['Parameters'][element]['UpperLimitCollision'] = False

            data = data_append(data, 'fit', element, sGraphOutput, flowerlimit, fupperlimit, perc[0], perc[1],
                               perc[2], perc[0]-perc[1], perc[2]-perc[1], confidence)

        # Derived parameters – results
        for origin in self.diStatResults['Results']:
            for name in self.diStatResults['Results'][origin]:
                vals = self.diStatResults['Results'][origin][name]
                perc = stats.scoreatpercentile(vals, percentiles)
                data = data_append(data, origin, name, '', None, None, perc[0], perc[1], perc[2], perc[0]-perc[1],
                                   perc[2]-perc[1], confidence)

        # Molgroups, internal storage of the results only
        self.fnProfilesStat(sparse=sparse, conf=percentiles)

        # SLD profiles, internal storage of the results only
        self.fnSLDProfilesStat(sparse=sparse, conf=percentiles)

        # Return Pandas dataframe of parameters and results
        return pandas.DataFrame(data)

    @staticmethod
    def fnCalcConfidenceLimits(data, method=1):
        # what follows is a set of three routines, courtesy to P. Kienzle, calculating
        # the shortest confidence interval

        def credible_interval(x, ci=0.95):
            """
            Find the credible interval covering the portion *ci* of the data.
            Returns the minimum and maximum values of the interval.
            *x* are samples from the posterior distribution.
            *ci* is the portion in (0,1], and defaults to 0.95.
            This function is faster if the inputs are already sorted.
            If *ci* is a vector, return a vector of intervals.
            """
            x.sort()
            if numpy.isscalar(ci):
                ci = [ci]

            # Simple solution: ci*N is the number of points in the interval, so
            # find the width of every interval of that size and return the smallest.
            result = [_find_interval(x, i) for i in ci]

            if len(ci) == 1:
                result = result[0]
            return result

        def _find_interval(x, ci):
            """
            Find credible interval ci in sorted, unweighted x
            """
            size = int(ci * len(x))
            if size > len(x) - 0.5:
                return x[0], x[-1]
            else:
                width = list(numpy.array(x[size:]) - numpy.array(x[:(-1 * size)]))
                idx = numpy.argmin(width)
                return x[idx], x[idx + size]

        # traditional method, taking percentiles of the entire distribution
        if method == 1:
            return [stats.scoreatpercentile(data, percentile) for percentile in [2.3, 15.9, 50, 84.1, 97.7]]
        # alternative method, starting from maximum of distribution
        elif method == 2:
            (histo, low_range, bin_size, _) = stats.histogram(data, numbins=int(len(data) / 5))

            # normalize histogram
            sumlist = sum(histo)
            for i in range(len(histo)):
                histo[i] = float(histo[i]) / sumlist

            maxindex = numpy.argmax(histo)
            print(maxindex, histo[maxindex])
            if histo[maxindex] == 1:
                return [data[0], data[0], data[0], data[0], data[0]]

            # calculate a smoother maximum value
            a = c = maxindex
            if 0 < maxindex < len(histo) - 1:
                a = maxindex - 1
                c = maxindex + 1
            maxindexsmooth = a * histo[a] + maxindex * histo[maxindex] + c * histo[c]
            maxindexsmooth = maxindexsmooth / (histo[a] + histo[maxindex] + histo[c])
            maxvaluesmooth = low_range + (maxindexsmooth + 0.5) * bin_size

            a = c = maxindex
            confidence = histo[maxindex]
            while confidence < 0.6827:
                if a > 0:
                    a -= 1
                    confidence += histo[a]
                if c < len(histo) - 1:
                    c += 1
                    confidence += histo[c]
            onesigmam = low_range + (a + 0.5) * bin_size
            onesigmap = low_range + (c + 0.5) * bin_size

            while confidence < 0.9545:
                if a > 0:
                    a -= 1
                    confidence += histo[a]
                if c < len(histo) - 1:
                    c += 1
                    confidence += histo[c]

            twosigmam = low_range + (a + 0.5) * bin_size
            twosigmap = low_range + (c + 0.5) * bin_size

            return [twosigmam, onesigmam, maxvaluesmooth, onesigmap, twosigmap]

        # shortest confidence interval method, NIST recommended
        else:
            twosigmam, twosigmap = credible_interval(data, 0.95)
            onesigmam, onesigmap = credible_interval(data, 0.68)
            reported = 0.5 * (onesigmam + onesigmap)
            return [twosigmam, onesigmam, reported, onesigmap, twosigmap]

    def corrected_bilayer_plot(self, plot_list=None, plot_uncertainties=None, plot=True):

        if plot_list is None:
            plot_list = ['substrate', 'siox', 'tether', 'innerhg', 'innerhc', 'outerhc', 'outerhg', 'protein', 'sum',
                         'water']
        if plot_uncertainties is None:
            plot_uncertainties = ['protein']

        # integrate over 1D array
        def fnIntegrate(axis, array, start, stop):
            idx_min = numpy.argmin(numpy.abs(axis - start))
            idx_max = numpy.argmin(numpy.abs(axis - stop)) + 1
            if idx_max > axis.shape[0]:
                idx_max = axis.shape[0]
            result = numpy.trapz(array[idx_min:idx_max], x=axis[idx_min:idx_max])
            return result

        # find maximum values and indizees of half-height points assuming unique solution and steady functions
        def fnMaximumHalfPoint(data):
            maximum = numpy.amax(data)
            point1 = False
            point2 = False
            hm1 = 0
            hm2 = 0
            for i in range(len(data)):
                if data[i] > (maximum / 2) and not point1:
                    point1 = True
                    hm1 = i
                if data[i] < (maximum / 2) and not point2:
                    point2 = True
                    hm2 = i - 1
                    i = len(data)
            return maximum, hm1, hm2

        def fnStat(area, name, diStat):
            diStat[name + '_msigma'] = numpy.percentile(area, q=16., axis=0)
            diStat[name] = numpy.percentile(area, q=50., axis=0)
            diStat[name + '_psigma'] = numpy.percentile(area, q=84., axis=0)

        # initialize Statistical Dictionary
        print('Initializing ...')
        lGroupList = ['substrate', 'siox', 'tether', 'innerhg', 'innerhc', 'outerhc', 'outerhg', 'protein',
                      'sum', 'water']
        diStat = {}
        for element in lGroupList:
            diStat[element] = None
            diStat[element + '_corr'] = None
            diStat[element + '_cvo'] = None
            diStat[element + '_corr_cvo'] = None

        keylist = list(diStat)
        for element in keylist:
            diStat[element + '_msigma'] = None
            diStat[element + '_psigma'] = None

        diIterations = {}
        for element in lGroupList:
            diIterations[element] = None
            diIterations[element + '_corr'] = None
            diIterations[element + '_cvo'] = None
            diIterations[element + '_corr_cvo'] = None

        # pull all relevant molgroups
        # headgroups are allready corrected for protein penetration (_corr) and will be copied over to the _corr
        # entries next

        i = 1
        mollist_innerhg = []
        mollist_innerhc = []
        mollist_outerhc = []
        mollist_outerhg = []
        while 'bilayer.headgroup1_' + str(i) in self.diStatResults['Molgroups']:
            mollist_innerhg.append('bilayer.headgroup1_' + str(i))
            mollist_innerhc.extend(['bilayer.methylene1_'+str(i), 'bilayer.methyl1_'+str(i)])
            mollist_outerhc.extend(['bilayer.methylene2_'+str(i), 'bilayer.methyl2_'+str(i)])
            mollist_outerhg.append('bilayer.headgroup2_' + str(i))
            i += 1

        print('Pulling all molgroups ...')
        print('  substrate ...')
        _, diIterations['substrate'], _, _ = self.molgroup_loader(['bilayer.substrate'])
        print('  siox ...')
        _, diIterations['siox'], _, _ = self.molgroup_loader(['bilayer.siox'])
        print('  tether ...')
        _, diIterations['tether'], _, _ = self.molgroup_loader(['bilayer.bME', 'bilayer.tetherg', 'bilayer.tether'])
        print('  innerhg ...')
        _, diIterations['innerhg'], __, __ = self.molgroup_loader(mollist_innerhg)
        print('  innerhc ...')
        _, diIterations['innerhc'], __, __ = self.molgroup_loader(mollist_innerhc)
        print('  outerhc ...')
        _, diIterations['outerhc'], __, __ = self.molgroup_loader(mollist_outerhc)
        print('  outerhg ...')
        _, diIterations['outerhg'], __, __ = self.molgroup_loader(mollist_outerhg)
        print('  protein ...')
        # save z-axis
        diStat['zaxis'], diIterations['protein'], __, __ = self.molgroup_loader(['protein'])

        diIterations['sum'] = numpy.zeros_like(diIterations['substrate'])
        diIterations['water'] = numpy.zeros_like(diIterations['substrate'])

        # shallow copies of the uncorrected data into the corrected dictionaries
        # and the values will be replaced by their modifications step by step
        for element in lGroupList:
            diIterations[element + '_corr'] = diIterations[element].copy()
            diIterations[element + '_cvo'] = diIterations[element].copy()
            diIterations[element + '_corr_cvo'] = diIterations[element].copy()

        # loop over all iterations and apply the corrections / calculations
        print('Applying corrections ...\n')
        for i in range(diIterations['substrate'].shape[0]):
            substrate = diIterations['substrate'][i]
            siox = diIterations['siox'][i]
            tether = diIterations['tether'][i]
            innerhg_corr = diIterations['innerhg_corr'][i]
            innerhc = diIterations['innerhc'][i]
            outerhc = diIterations['outerhc'][i]
            outerhg_corr = diIterations['outerhg_corr'][i]
            protein = diIterations['protein'][i]
            axis = diStat['zaxis']

            hc = innerhc + outerhc
            # this is the sum as used for the joining procedure
            sum = substrate + siox + tether + innerhg_corr + hc + outerhg_corr

            areaperlipid, _, _ = fnMaximumHalfPoint(substrate)
            maxbilayerarea, _, _ = fnMaximumHalfPoint(hc)
            # vf_bilayer = maxbilayerarea/areaperlipid

            # recuperate the non-corrected headgroup distributions that were not saved to file by the fit
            # by reversing the multiplication based on the amount of replaced hc material
            __, hc1_hm1, hc1_hm2 = fnMaximumHalfPoint(innerhc)
            __, hc2_hm1, hc2_hm2 = fnMaximumHalfPoint(outerhc)
            hg1ratio = fnIntegrate(axis, protein, axis[hc1_hm1], axis[hc1_hm2]) \
                       / fnIntegrate(axis, innerhc, axis[hc1_hm1], axis[hc1_hm2])
            hg2ratio = fnIntegrate(axis, protein, axis[hc2_hm1], axis[hc2_hm2]) \
                       / fnIntegrate(axis, outerhc, axis[hc2_hm1], axis[hc2_hm2])
            innerhg = innerhg_corr / (1 - hg1ratio)
            diIterations['innerhg'][i] = numpy.copy(innerhg)
            outerhg = outerhg_corr / (1 - hg2ratio)
            diIterations['outerhg'][i] = numpy.copy(outerhg)

            # prepare arrays for correction
            innerhc_corr = numpy.copy(innerhc)
            outerhc_corr = numpy.copy(outerhc)

            # correct the hc profiles due to protein penetration
            for j in range(len(protein)):
                if sum[j] + protein[j] > maxbilayerarea:
                    if (innerhc[j] + outerhc[j]) > 0:
                        excess = sum[j] + protein[j] - maxbilayerarea
                        if excess > (innerhc[j] + outerhc[j]):
                            excess = (innerhc[j] + outerhc[j])
                        # print (innerhc[i]+outerhc[i]) > 0, i
                        # print 'first' , innerhc_corr[i], excess, innerhc[i], outerhc[i],
                        # excess*innerhc[i]/(innerhc[i]+outerhc[i])
                        innerhc_corr[j] -= excess * innerhc[j] / (innerhc[j] + outerhc[j])
                        # print 'second' , outerhc_corr[i], excess, innerhc[i], outerhc[i],
                        # excess*outerhc[i]/(innerhc[i]+outerhc[i])
                        outerhc_corr[j] -= excess * outerhc[j] / (innerhc[j] + outerhc[j])

            # update dictionary entries for later statistics
            diIterations['innerhc_corr'][i] = numpy.copy(innerhc_corr)
            diIterations['outerhc_corr'][i] = numpy.copy(outerhc_corr)
            sum_corr = substrate + siox + tether + innerhg_corr + innerhc_corr + outerhc_corr + outerhg_corr + protein
            diIterations['sum_corr'][i] = numpy.copy(sum_corr)

            # this is the truly non-corrected sum, different from the previous sum used for joining
            sum = substrate + siox + tether + innerhg + innerhc + outerhc + outerhg + protein
            diIterations['sum'][i] = numpy.copy(sum)
            water_corr = areaperlipid - sum_corr
            diIterations['water_corr'][i] = numpy.copy(water_corr)
            water = areaperlipid - sum
            diIterations['water'][i] = numpy.copy(water)

            # calculate volume occupancy distributions by division by the area per lipid
            for element in lGroupList:
                diIterations[element + '_cvo'][i] = diIterations[element][i] / areaperlipid
                diIterations[element + '_corr_cvo'][i] = diIterations[element + '_corr'][i] / areaperlipid

        # calculate the statisics
        print('Calculating statistics ...\n')
        for element in lGroupList:
            if element != 'zaxis':
                fnStat(diIterations[element], element, diStat)
                fnStat(diIterations[element + '_corr'], element + '_corr', diStat)
                fnStat(diIterations[element + '_cvo'], element + '_cvo', diStat)
                fnStat(diIterations[element + '_corr_cvo'], element + '_corr_cvo', diStat)

        print('Saving data to bilayerplotdata.dat ...\n')
        self.Interactor.fnSaveSingleColumns('bilayerplotdata.dat', diStat)

        if plot:
            fig, ax = plt.subplots()
            for gp in plot_list:
                if gp + '_corr_cvo' in diStat:
                    zaxis = diStat['zaxis']
                    area = diStat[gp+'_corr_cvo']
                    ax.plot(zaxis, area, label=gp)
                    if gp in plot_uncertainties:
                        msigma = diStat[gp + '_corr_cvo_msigma']
                        psigma = diStat[gp + '_corr_cvo_psigma']
                        ax.fill_between(zaxis, msigma, psigma, alpha=0.3)
                elif gp in self.diStatResults['Molgroups']:
                    zaxis = self.diStatResults['Molgroups'][gp]['zaxis']
                    area = self.diStatResults['Molgroups'][gp]['median area']
                    ax.plot(zaxis, area, label=gp)
                    if gp in plot_uncertainties:
                        msigma = self.diStatResults['Molgroups'][gp]['msigma area']
                        psigma = self.diStatResults['Molgroups'][gp]['psigma area']
                        ax.fill_between(zaxis, msigma, psigma, alpha=0.3)

            ax.legend(loc="upper right")
            plt.xlabel("Distance (Å)")
            plt.ylabel("Area (Å)")
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", labelsize=10)
            ax.tick_params(bottom=True, top=True, left=True, right=True, which="both")
            # plt.xlim(0, 100)
            # plt.xticks(numpy.arange(-35, 36, 5.0))
            plt.grid(True, which='Both')
            fig.patch.set_facecolor('white')
            ax.figure.set_size_inches(10, 6.66)
            plt.savefig(os.path.join(self.spath, self.mcmcpath, 'cvo_corr.png'), facecolor="white")
            plt.show()

    def draw_sample(self, parlist=None):
        if self.diStatResults == {} or parlist is None:
            return None
        iteration = numpy.random.randint(0, self.diStatResults['NumberOfStatValues'])
        retval = []
        for par in parlist:
            retval.append(self.diStatResults['Parameters'][par]['Values'][iteration])
        return retval

    def fnGetChiSq(self):  # export chi squared
        return self.chisq

    def fnGetParameterValue(self, sname):  # export absolute parameter value
        return self.diParameters[sname]['value']  # for given name

    def fnGetSortedParNames(self):  # return a list of sorted parameter
        litest = list(self.diParameters.keys())
        litest = sorted(litest, key=lambda keyitem: self.diParameters[keyitem]['number'])
        return litest

    def fnGetSortedParValues(self):  # the same as above but it returns
        litest = list(self.diParameters.keys())  # a list of sorted parameter values
        litest = sorted(litest, key=lambda keyitem: self.diParameters[keyitem]['number'])
        lvalue = []
        for parameter in litest:
            lvalue.append(str(self.diParameters[parameter]['value']))
        return lvalue

    def fnLoadAndPrintPar(self, sPath='./'):
        self.fnLoadParameters()
        self.fnLoadCovar(sPath)
        self.fnPrintPar()

    def fnLoadCovar(self, sPath):
        """
        loads a covariance matrix, calculates the errors and stores it into self.diParameters
        The function fnLoadParameters must have been already carried out
        """
        if path.isfile(f'{sPath}covar.dat'):
            with open(f'{sPath}covar.dat') as file:
                data = file.readlines()
            for i in range(len((data[1]).split())):  # number of columns in second row
                for parameter in list(self.diParameters.keys()):  # search for parameter number and
                    if self.diParameters[parameter]['number'] == i:  # retrieve value
                        fvalue = self.diParameters[parameter]['value']
                        ferror = float((data[i + 1]).split()[i])  # the first row contains a comment
                        if ferror < 0:
                            ferror = 0
                        ferror = sqrt(ferror) * fvalue  # calculate error
                        self.diParameters[parameter]['error'] = ferror
                        break
        else:
            for parameter in list(self.diParameters.keys()):
                self.diParameters[parameter]['error'] = float(0)  # fill all errors with zero

    @staticmethod
    def fnLoadObject(sFileName):
        import pickle
        with open(sFileName, "rb") as file:
            Object = pickle.load(file)
        return Object

    def fnLoadParameters(self):
        if self.diParameters == {}:
            self.diParameters, self.chisq = self.Interactor.fnLoadParameters()
        return self.diParameters

    def fnLoadStatData(self, sparse=0):
        self.fnLoadParameters()
        if self.diStatResults != {}:
            return

        try:
            self.diStatResults = self.fnLoadObject(os.path.join(self.spath, self.mcmcpath, 'StatDataPython.dat'))
            print('Loaded statistical data from StatDataPython.dat')
            return
        except IOError:
            print('No StatDataPython.dat.')
            print('Recreate statistical data from sErr.dat.')

        self.diStatResults = self.Interactor.fnLoadStatData(sparse)
        # cycle through all parameters, determine length of the longest parameter name for displaying
        iMaxParameterNameLength = 0
        for parname in list(self.diStatResults['Parameters'].keys()):
            if len(parname) > iMaxParameterNameLength:
                iMaxParameterNameLength = len(parname)
        self.diStatResults['MaxParameterLength'] = iMaxParameterNameLength
        self.diStatResults['NumberOfStatValues'] = \
            len(self.diStatResults['Parameters'][list(self.diStatResults['Parameters'].keys())[0]]['Values'])

        # Recreates profile and fit data associated with parameter stats
        if self.Interactor.problem is not None:
            problem = self.Interactor.problem
        else:
            problem = self.Interactor.fnRestoreFitProblem()

        j = 0
        self.diStatResults['nSLDProfiles'] = []
        self.diStatResults['Molgroups'] = {}
        self.diStatResults['Results'] = {}

        for iteration in range(self.diStatResults['NumberOfStatValues']):
            try:
                liParameters = list(self.diParameters.keys())

                # sort by number of appereance in setup file
                liParameters = sorted(liParameters, key=lambda keyitem: self.diParameters[keyitem]['number'])
                bConsistency = set(liParameters).issubset(self.diStatResults['Parameters'].keys())
                if not bConsistency:
                    raise RuntimeError('Statistical error data and setup file do not match')

                p = []
                for parameter in liParameters:
                    val = self.diStatResults['Parameters'][parameter]['Values'][iteration]
                    p.append(val)
                problem.setp(p)
                problem.model_update()

                # TODO: By calling .chisq() I currently force an update of the BLM function. There must be a better
                #   way, also implement which contrast to use for pulling groups garefl based code should save a
                #   mol.dat automatically on updating the model
                if 'models' in dir(problem):
                    for M in problem.models:
                        M.chisq()
                else:
                    problem.chisq()

                # Recreate Molgroups and Derived Results
                self.diMolgroups, self.diResults = self.Interactor.fnLoadMolgroups(problem)

                if self.diMolgroups is not None and self.diResults is not None:
                    # Store Molgroups
                    # self.diStatResults['Molgroups'].append(self.diMolgroups)
                    for name in self.diMolgroups:
                        if name not in self.diStatResults['Molgroups']:
                            self.diStatResults['Molgroups'][name] = {}
                        for entry in self.diMolgroups[name]:
                            if entry == 'zaxis':
                                # store z-axis only once
                                if entry not in self.diStatResults['Molgroups'][name]:
                                    self.diStatResults['Molgroups'][name][entry] = self.diMolgroups[name][entry]
                            else:
                                if entry not in self.diStatResults['Molgroups'][name]:
                                    self.diStatResults['Molgroups'][name][entry] = []
                                self.diStatResults['Molgroups'][name][entry].append(self.diMolgroups[name][entry])

                    # Store Derived Results
                    # origin is the name of the object that provided a result with a certain name
                    for origin in self.diResults:
                        if origin not in self.diStatResults['Results']:
                            self.diStatResults['Results'][origin] = {}
                        for name in self.diResults[origin]:
                            if name not in self.diStatResults['Results'][origin]:
                                self.diStatResults['Results'][origin][name] = []
                            self.diStatResults['Results'][origin][name].append(self.diResults[origin][name])

                # distinguish between FitProblem and MultiFitProblem
                if 'models' in dir(problem):
                    for M in problem.models:
                        if not isinstance(M.fitness, bumps.curve.Curve):
                            z, rho, irho = self.Interactor.fnRestoreSmoothProfile(M)
                            self.diStatResults['nSLDProfiles'].append((z, rho, irho))
                            # only report SLD profile for first model
                            break
                else:
                    z, rho, irho = self.Interactor.fnRestoreSmoothProfile(problem)
                    self.diStatResults['nSLDProfiles'].append((z, rho, irho))


            finally:
                j += 1

        # save stat data to disk, if flag is set
        if self.save_stat_data:
            self.fnSaveObject(self.diStatResults, os.path.join(self.spath, self.mcmcpath, 'StatDataPython.dat'))

    def fnPrintPar(self):
        # prints parameters and their errors from the covariance matrix onto the screen

        litest = list(self.diParameters.keys())
        litest = sorted(litest, key=lambda keyitem: self.diParameters[keyitem]['number'])
        for parameter in litest:
            fRange = (self.diParameters[parameter]['upperlimit']
                      - self.diParameters[parameter]['lowerlimit'])
            fLowLim = self.diParameters[parameter]['lowerlimit']
            fValue = self.diParameters[parameter]['value']
            sRangeIndicator = ''
            for i in range(10):  # creates the par range ascii overview
                if ((fValue >= float(i) / 10 * fRange + fLowLim) and
                        (fValue < float(i + 1) / 10 * fRange + fLowLim)):
                    sRangeIndicator += '|'
                else:
                    if (fValue == float(i + 1) / 10 * fRange + fLowLim) and (i == 9):
                        sRangeIndicator += '|'
                    else:
                        sRangeIndicator += '.'
            print('%2i %25s  %s %15g +/- %g in [%g,%g]' % (self.diParameters[parameter]['number'],
                                                           parameter, sRangeIndicator, fValue,
                                                           self.diParameters[parameter]['error'],
                                                           fLowLim, self.diParameters[parameter]['upperlimit']))
        print('Chi squared: %g' % self.chisq)

    def fnProfilesStat(self, sparse=0, conf=(18., 50., 82.)):
        """
        Summarizes the statistical data for all molecular groups from the MCMC and produces median and ± sigma
        profiles of the area, SL, and SLD for each. Results are stored in the internal self.diStatResults dictionary
        under self.diStatResults['Molgroups'][group]['median area', 'median sl', 'median sld', 'psigma area',
        'psigma sl', 'psigma sld', 'msigma area', 'msigma sl', 'msigma sld'].

        :param sparse: 0 < sparse < 1, fraction of statistical data from the MCMC that is used the summary
                           sparse > 1, number of iterations from the MCMC used for summary
        :param conf: tuple of percentiles (<100) for median and confidence limits (lower, median, higher)

        :return: no return value
        """
        self.fnLoadStatData(sparse)

        if 'Molgroups' not in self.diStatResults.keys():
            return

        c1 = conf[0]
        c2 = conf[1]
        c3 = conf[2]

        for group in self.diStatResults['Molgroups']:
            median_area = numpy.percentile(self.diStatResults['Molgroups'][group]['area'], c2, axis=0)
            psigma_area = numpy.percentile(self.diStatResults['Molgroups'][group]['area'], c3, axis=0)
            msigma_area = numpy.percentile(self.diStatResults['Molgroups'][group]['area'], c1, axis=0)
            median_sl = numpy.percentile(self.diStatResults['Molgroups'][group]['sl'], c2, axis=0)
            psigma_sl = numpy.percentile(self.diStatResults['Molgroups'][group]['sl'], c3, axis=0)
            msigma_sl = numpy.percentile(self.diStatResults['Molgroups'][group]['sl'], c1, axis=0)
            median_sld = numpy.percentile(self.diStatResults['Molgroups'][group]['sld'], c2, axis=0)
            psigma_sld = numpy.percentile(self.diStatResults['Molgroups'][group]['sld'], c3, axis=0)
            msigma_sld = numpy.percentile(self.diStatResults['Molgroups'][group]['sld'], c1, axis=0)

            self.diStatResults['Molgroups'][group]['median area'] = median_area
            self.diStatResults['Molgroups'][group]['psigma area'] = psigma_area
            self.diStatResults['Molgroups'][group]['msigma area'] = msigma_area
            self.diStatResults['Molgroups'][group]['median sl'] = median_sl
            self.diStatResults['Molgroups'][group]['psigma sl'] = psigma_sl
            self.diStatResults['Molgroups'][group]['msigma sl'] = msigma_sl
            self.diStatResults['Molgroups'][group]['median sld'] = median_sld
            self.diStatResults['Molgroups'][group]['psigma sld'] = psigma_sld
            self.diStatResults['Molgroups'][group]['msigma sld'] = msigma_sld

    def fnPullMolgroup(self, liMolgroupNames, sparse=0, verbose=True):
        """
        Calls Function that recreates statistical data and extracts only area and nSL profiles for
        submolecular groups whose names are given in liMolgroupNames. Those groups
        are added for each iteration and a file pulledmolgroups.dat is created.
        A statistical analysis area profile containing the median, sigma, and
        2 sigma intervals are put out in pulledmolgroupsstat.dat.
        Save results to file
        """
        diarea, dinsl, dinsld = self.molgroup_loader(liMolgroupNames, sparse, verbose=verbose)
        diStat = dict(zaxis=[], m2sigma_area=[], msigma_area=[], median_area=[], psigma_area=[], p2sigma_area=[],
                      m2sigma_nsl=[], msigma_nsl=[], median_nsl=[], psigma_nsl=[], p2sigma_nsl=[],
                      m2sigma_nsld=[], msigma_nsld=[], median_nsld=[], psigma_nsld=[], p2sigma_nsld=[])

        for i in range(len(diarea[list(diarea.keys())[0]])):
            liOnePosition = [iteration[i] for key, iteration in diarea.items() if key != 'zaxis']
            stat = self.fnCalcConfidenceLimits(liOnePosition, method=1)
            diStat['zaxis'].append(str(diarea['zaxis'][i]))
            diStat['m2sigma_area'].append(stat[0])
            diStat['msigma_area'].append(stat[1])
            diStat['median_area'].append(stat[2])
            diStat['psigma_area'].append(stat[3])
            diStat['p2sigma_area'].append(stat[4])

            liOnePosition = [iteration[i] for key, iteration in dinsl.items() if key != 'zaxis']
            stat = self.fnCalcConfidenceLimits(liOnePosition, method=1)
            diStat['m2sigma_nsl'].append(stat[0])
            diStat['msigma_nsl'].append(stat[1])
            diStat['median_nsl'].append(stat[2])
            diStat['psigma_nsl'].append(stat[3])
            diStat['p2sigma_nsl'].append(stat[4])

            liOnePosition = [iteration[i] for key, iteration in dinsld.items() if key != 'zaxis']
            stat = self.fnCalcConfidenceLimits(liOnePosition, method=1)
            diStat['m2sigma_nsld'].append(stat[0])
            diStat['msigma_nsld'].append(stat[1])
            diStat['median_nsld'].append(stat[2])
            diStat['psigma_nsld'].append(stat[3])
            diStat['p2sigma_nsld'].append(stat[4])

        self.Interactor.fnSaveSingleColumns(self.mcmcpath + '/pulledmolgroups_area.dat', diarea)
        self.Interactor.fnSaveSingleColumns(self.mcmcpath + '/pulledmolgroups_nsl.dat', dinsl)
        self.Interactor.fnSaveSingleColumns(self.mcmcpath + '/pulledmolgroups_nsld.dat', dinsld)
        self.Interactor.fnSaveSingleColumns(self.mcmcpath + '/pulledmolgroupsstat.dat', diStat)

    def molgroup_loader(self, liMolgroupNames, sparse=0):
        """
        Function recreates statistical data and extracts only area and nSL profiles for
        submolecular groups whose names are given in liMolgroupNames. Those groups
        are added for each iteration and a file pulledmolgroups.dat is created.
        A statistical analysis area profile containing the median, sigma, and
        2 sigma intervals are put out in pulledmolgroupsstat.dat.
        """

        if self.diStatResults == {}:
            self.fnLoadStatData()

        zaxis = numpy.array(self.diStatResults['Molgroups'][list(self.diStatResults['Molgroups'].keys())[0]]['zaxis'])
        area = numpy.zeros_like(self.diStatResults['Molgroups'][list(self.diStatResults['Molgroups'].keys())[0]]
                                ['area'])
        sl = numpy.zeros_like(area)
        sld = numpy.zeros_like(area)

        for gp in liMolgroupNames:
            if gp in self.diStatResults['Molgroups']:
                area_gp = numpy.array(self.diStatResults['Molgroups'][gp]['area'])
                area += area_gp
                sl_gp = numpy.array(self.diStatResults['Molgroups'][gp]['sl'])
                sl += sl_gp
                sld += numpy.array(self.diStatResults['Molgroups'][gp]['sld']) * area_gp
            else:
                print(f'Molecular group {gp} does not exist.')

        sld = sld / area

        return zaxis, area, sl, sld

    def fnRestoreFit(self):
        self.Interactor.fnRestoreFit()

    def fnRunFit(self, burn=2000, steps=500, batch=False, resume=False):
        path1 = os.path.join(self.spath, self.mcmcpath)
        if os.path.isfile(os.path.join(path1, "sErr.dat")):
            os.remove(os.path.join(path1, "sErr.dat"))
        if os.path.isfile(os.path.join(path1,  "isErr.dat")):
            os.remove(os.path.join(path1,  "isErr.dat"))
        if os.path.isfile(os.path.join(path1,  "StatDataPython.dat")):
            os.remove(os.path.join(path1,  "StatDataPython.dat"))
        self.Interactor.fnRunMCMC(burn, steps, batch=batch, resume=resume)

    @staticmethod
    def fnSaveObject(save_object, sFileName):
        import pickle

        with open(sFileName, "wb") as file:
            pickle.dump(save_object, file)

    def fnSimulateData(self, basefilename='sim.dat', liConfigurations=None, qmin=None, qmax=None, qrangefromfile=False,
                       t_total=None, mode='water', lambda_min=0.1, verbose=True, simpar=None, save_file=True,
                       average=False):
        """
        Simulates scattering based on a parameter file called simpar.dat
        requires a ready-to-go fit whose fit parameters are modified and fixed
        The basename can refer to a set of data files with integer indizes before the suffix
        """

        # Load Parameters
        self.diParameters, _ = self.Interactor.fnLoadParameters()

        liParameters = list(self.diParameters.keys())
        liParameters = sorted(liParameters, key=lambda keyitem: self.diParameters[keyitem]['number'])

        if simpar is None:
            # the file simpar.dat contains the parameter values to be simulated
            # this could be done fileless
            simpar = pandas.read_csv(self.spath + '/simpar.dat', sep='\s+', header=None, names=['par', 'value'],
                                     skip_blank_lines=True, comment='#')
        else:
            # simpar is provided as a dataframe with the appropriate shape
            simpar.columns = ['par', 'value']

        if verbose:
            print(simpar)
            print(liConfigurations)

        diModelPars = {}
        for parameter in liParameters:
            diModelPars[parameter] = simpar[simpar.par == parameter].iloc[0][1]
        # load all data files into a list of Pandas dataframes
        # each element is itself a list of [comments, simdata]
        liData = self.Interactor.fnLoadData(basefilename)
        liData = self.Interactor.fnSimulateDataPlusErrorBars(liData, diModelPars, simpar=simpar,
                                                             basefilename=basefilename,
                                                             liConfigurations=liConfigurations, qmin=qmin,
                                                             qmax=qmax, qrangefromfile=qrangefromfile,
                                                             lambda_min=lambda_min, mode=mode, t_total=t_total,
                                                             average=average)

        # always save the file since it has been modified in place before
        # TODO: one could make this more consistent and remove save_file from function signature
        self.Interactor.fnSaveData(basefilename, liData)
        return liData

    def fnSLDProfilesStat(self, sparse=0, conf=(18., 50., 82.)):
        """
        Summarizes the statistical data for all SLD profiles from the MCMC and produces median and ± sigma
        profiles of the rho and irho for each. Results are stored in the internal self.diStatResults dictionary
        under self.diStatResults['SLD']['median rho', 'msigma rho', 'psigma rho', 'median irho', 'msigma irho',
        'psigma irho'].

        :param sparse: 0 < sparse < 1, fraction of statistical data from the MCMC that is used the summary
                           sparse > 1, number of iterations from the MCMC used for summary
        :param conf: tuple of percentiles (<100) for median and confidence limits (lower, median, higher)
        :return: no return value
        """
        self.fnLoadStatData(sparse)

        if 'nSLDProfiles' not in self.diStatResults.keys():
            return

        # identify longest z-axis in stat data
        maxlength = 0
        maxpos = 0
        for i, dataset in enumerate(self.diStatResults['nSLDProfiles']):
            length = len(dataset[0])
            if length > maxlength:
                maxlength = length
                maxpos = i

        if maxlength <= 0:
            return

        numdatasets = len(self.diStatResults['nSLDProfiles'])
        profiles = numpy.empty([numdatasets, 3, maxlength])

        # interpolate profiles onto max length, this assumes that the longest axis also has the
        # smallest and largest values, which might be false, but should be inconsequential in most
        # cases
        for i, dataset in enumerate(self.diStatResults['nSLDProfiles']):
            profiles[i, 0] = self.diStatResults['nSLDProfiles'][maxpos][0]
            profiles[i, 1] = numpy.interp(profiles[i, 0], dataset[0], dataset[1])
            profiles[i, 2] = numpy.interp(profiles[i, 0], dataset[0], dataset[2])

        rho = profiles[:, 1]
        irho = profiles[:, 2]
        c1 = conf[0]
        c2 = conf[1]
        c3 = conf[2]

        self.diStatResults['SLD'] = {}
        self.diStatResults['SLD']['z'] = profiles[0, 0]
        self.diStatResults['SLD']['msigma rho'] = numpy.percentile(rho, c1, axis=0)
        self.diStatResults['SLD']['median rho'] = numpy.percentile(rho, c2, axis=0)
        self.diStatResults['SLD']['psigma rho'] = numpy.percentile(rho, c3, axis=0)
        self.diStatResults['SLD']['msigma irho'] = numpy.percentile(irho, c1, axis=0)
        self.diStatResults['SLD']['median irho'] = numpy.percentile(irho, c2, axis=0)
        self.diStatResults['SLD']['psigma irho'] = numpy.percentile(irho, c3, axis=0)

