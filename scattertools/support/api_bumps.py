from __future__ import print_function
from os import path
from random import seed, random
from re import VERBOSE, IGNORECASE, compile
from sys import stdout

import matplotlib
from matplotlib import pyplot as plt

from bumps.cli import save_best
from bumps.mapper import MPMapper
from bumps.fitters import FitDriver, DreamFit, MPFit

import numpy
import shutil
import glob
import os

from scattertools.support import api_base


class CBumpsAPI(api_base.CBaseAPI):
    def __init__(self, spath=".", mcmcpath=".", runfile="", state=None, problem=None, load_state=True):
        super().__init__(spath, mcmcpath, runfile)
        if load_state:
            self.state = self.fnRestoreState() if state is None else state
        self.problem = self.fnRestoreFitProblem() if problem is None else problem

    def fnBackup(self, origin=None, target=None):
        if origin is None:
            origin = self.spath
        if target is None:
            target = self.spath + '/rsbackup'
        if not path.isdir(target):
            os.mkdir(target)
        for file in glob.glob(origin + r'/*.dat'):
            shutil.copy(file, target)
        for file in glob.glob(origin + r'/*.py'):
            shutil.copy(file, target)
        for file in glob.glob(origin + r'/*.pyc'):
            shutil.copy(file, target)

    def fnLoadMCMCResults(self):
        # load Parameter
        if self.diParameters == {}:
            self.diParameters, _ = self.fnLoadParameters()
        lParName = self.diParameters.keys()

        # parvars is a list of variables(parameters) to return for each point
        parvars = [i for i in range(len(lParName))]
        draw = self.state.draw(1, parvars, None)
        points = draw.points
        logp = draw.logp

        return points, lParName, logp

    def fnLoadMolgroups(self, problem=None):
        if problem is None or not hasattr(problem, 'results') or not hasattr(problem, 'moldat'):
            return None, None

        diMolgroups = {}
        diResults = problem.results
        moldict = problem.moldat

        for group in moldict:
            tdata = (moldict[group]['header']).split()  # read header that contains molgroup data
            diMolgroups[tdata[1]] = {}
            diMolgroups[tdata[1]].update({'headerdata': {}})
            diMolgroups[tdata[1]]['headerdata'].update({'Type': tdata[0]})
            diMolgroups[tdata[1]]['headerdata'].update({'ID': tdata[1]})
            for j in range(2, len(tdata), 2):
                diMolgroups[tdata[1]]['headerdata'].update({tdata[j]: tdata[j + 1]})

            zax = moldict[group]['zaxis']
            areaax = moldict[group]['area']
            nslax = moldict[group]['sl']
            sldax = moldict[group]['sld']
            diMolgroups[tdata[1]].update({'zaxis': zax, 'area': areaax, 'sl': nslax, 'sld': sldax})

        return diMolgroups, diResults

    # LoadStatResults returns a list of variable names, a logP array, and a numpy.ndarray
    # [values,var_numbers].
    def fnLoadParameters(self):
        if self.state is None:
            self.state = self.fnRestoreState()

        if self.state is not None:
            p = self.state.best()[0]
            self.problem.setp(p)
        else:
            p = self.problem.getp()
            print('Parameter directory does not contain best fit because state not loaded.')
            print('Parameter values are from initialization of the model.')

        # from bumps.cli import load_best
        # load_best(problem, os.path.join(self.mcmcpath, self.runfile) + '.par')

        self.problem.model_update()
        overall = self.problem.fitness()
        pnamekeys = self.problem.labels()

        # Do not accept parameter names with spaces, replace with underscore
        for i in range(len(pnamekeys)):
            pnamekeys[i] = pnamekeys[i].replace(" ", "_")

        for element in pnamekeys:
            self.diParameters[element] = {}
        bounds = self.problem.bounds()

        for key in pnamekeys:
            parindex = pnamekeys.index(key)
            self.diParameters[key]["number"] = parindex
            self.diParameters[key]["lowerlimit"] = float(bounds[0][parindex])
            self.diParameters[key]["upperlimit"] = float(bounds[1][parindex])
            self.diParameters[key]["value"] = float(p[parindex])
            self.diParameters[key]["relval"] = (
                self.diParameters[key]["value"] - self.diParameters[key]["lowerlimit"]
            ) / (
                self.diParameters[key]["upperlimit"]
                - self.diParameters[key]["lowerlimit"]
            )
            # TODO: Do we still need this? Set to dummy value
            self.diParameters[key]["variable"] = key
            # TODO: Do we still need this? Would have to figure out how to get the confidence limits from state
            self.diParameters[key]["error"] = 0.01

        return self.diParameters, overall

    def fnLoadStatData(self, dSparse=0, rescale_small_numbers=True, skip_entries=None):
        if skip_entries is None:
            skip_entries = []

        if path.isfile(os.path.join(self.spath, self.mcmcpath, "sErr.dat")) or \
                path.isfile(os.path.join(self.spath, self.mcmcpath, "isErr.dat")):
            diStatRawData = self.fnLoadsErr()
        else:
            points, lParName, logp = self.fnLoadMCMCResults()
            diStatRawData = {"Parameters": {}}
            diStatRawData["Parameters"]["Chisq"] = {}
            # TODO: Work on better chisq handling
            diStatRawData["Parameters"]["Chisq"]["Values"] = []
            for parname in lParName:
                diStatRawData["Parameters"][parname] = {}
                diStatRawData["Parameters"][parname]["Values"] = []

            seed()
            for j in range(len(points[:, 0])):
                if dSparse == 0 or (dSparse > 1 and j < dSparse) or (1 >= dSparse > random()):
                    diStatRawData["Parameters"]["Chisq"]["Values"].append(logp[j])
                    for i, parname in enumerate(lParName):
                        diStatRawData["Parameters"][parname]["Values"].append(points[j, i])

            self.fnSaveSingleColumnsFromStatDict(os.path.join(self.spath, self.mcmcpath, "sErr.dat"),
                                                 diStatRawData["Parameters"], skip_entries)

        return diStatRawData

    # deletes the backup directory after restoring it
    def fnRemoveBackup(self, origin=None, target=None):
        if origin is None:
            origin = self.spath
        if target is None:
            target = self.spath + '/rsbackup'
        if path.isdir(target):
            self.fnRestoreBackup(origin, target)
            shutil.rmtree(target)

    def fnReplaceParameterLimitsInSetup(self, sname, flowerlimit, fupperlimit, modify=None):
        """
        Scans self.runfile file for parameter with name sname and replaces the
        lower and upper fit limits by the given values If an initialization value is given as part of the Parameter()
        function, it will be replaced as well. The value= argument should follow the name= argument in Parameter()
        :param sname: name of the parameter
        :param flowerlimit: lower fit bound
        :param fupperlimit: upper fit bound
        :param modify: default is None, 'add' add parameter with name and limits, 'remove' remove parameter from script
        :return: None
        """

        file = open(os.path.join(self.spath, self.runfile) + '.py', 'r+')
        data = file.readlines()
        file.close()
        smatch = compile(r"(.*?Parameter.*?name=\'"+sname+"[\"\'].+?=).+?(\).+?range\().+?(,).+?(\).*)",
                         IGNORECASE | VERBOSE)
        # version when .range() is present but no parameter value is provided
        smatch2 = compile(r"(.*?\."+sname+'\.range\().+?(,).+?(\).*)', IGNORECASE | VERBOSE)
        newdata = []
        found_match = False
        for line in data:
            # apply version 1 for general case
            newline = smatch.sub(r'\1 ' + str(0.5*(flowerlimit+fupperlimit)) + r'\2 ' + str(flowerlimit) + r'\3 ' +
                                 str(fupperlimit) + r'\4', line)
            # apply version 2 to catch both cases, potentially redundant for limits
            newline = smatch2.sub(r'\1 ' + str(flowerlimit) + r'\2 ' + str(fupperlimit) + r'\3', newline)

            mstring1 = smatch.match(line)
            mstring2 = smatch2.match(line)
            if mstring1 or mstring2:
                found_match = True
                # check if instruction is commented out, but we want to add it to the script
                # does not support comment blocks
                if modify == 'add':
                    newline = newline.strip()
                    while len(newline) > 0 and newline[0] == '#':
                        newline = newline[1:]
                    newline = newline.strip()
                    newline = newline + '\n'
            # remove par if match found and flag set
            if not (modify == 'remove' and (mstring1 or mstring2)):
                newdata.append(newline)

        if modify == 'add' and not found_match:
            insertpoint = None
            codestr = ''
            smatch3 = compile(r".*?Experiment.*?model.*?=(.*?)\).*", IGNORECASE | VERBOSE)
            for i, line in reversed(list(enumerate(newdata))):
                mstring = smatch3.match(line)
                if mstring:
                    insertpoint = i
                    modelname = mstring.groups()[0]
                    modelname = modelname.strip()
                    codestr = modelname + '.' + sname + '.range(' + str(flowerlimit) + ', ' + str(fupperlimit) + ')\n'

            if insertpoint is not None and insertpoint > 0:
                if newdata[insertpoint-1].isspace():
                    insertpoint -= 1
                newdata.insert(insertpoint, codestr)

        file = open(os.path.join(self.spath, self.runfile) + '.py', 'w')
        file.writelines(newdata)
        file.close()

    # copies all files from the backup directory (target) to origin
    def fnRestoreBackup(self, origin=None, target=None):
        if origin is None:
            origin = self.spath
        if target is None:
            target = self.spath + '/rsbackup'
        if path.isdir(target):
            for file in glob.glob(target + r'/*.*'):
                shutil.copy(file, origin)

    def fnRestoreFit(self):
        self.problem = self.fnRestoreFitProblem()
        self.state = self.fnRestoreState()
        if self.state is not None:
            # repopulate state with best fit
            p = self.state.best()[0]
            self.problem.setp(p)

    def fnRestoreFitProblem(self):
        from bumps.fitproblem import load_problem

        if path.isfile(os.path.join(self.spath, self.runfile + ".py")):
            problem = load_problem(os.path.join(self.spath, self.runfile + ".py"))
        else:
            print("No file: " + os.path.join(self.spath, self.runfile + ".py"))
            print("No problem to reload.")
            problem = None
        return problem

    def fnRestoreState(self):
        import bumps.dream.state
        fulldir = os.path.join(self.spath, self.mcmcpath)
        if path.isfile(os.path.join(fulldir, self.runfile) + '.py') and path.isfile(os.path.join(fulldir, self.runfile)
                                                                                    + '-chain.mc.gz'):
            state = bumps.dream.state.load_state(os.path.join(fulldir, self.runfile))
            state.mark_outliers()  # ignore outlier chains
        else:
            print("No file: " + os.path.join(fulldir, self.runfile) + '.py')
            print("No state to reload.")
            state = None
        return state

    def fnRestoreSmoothProfile(self, M):
        # TODO: Decide what and if to return SLD profile for Bumps fits
        # Returns currently profile for zeroth model if multiproblem fit
        z, rho, irho = M.sld, [], []
        return z, rho, irho

    def fnRunMCMC(self, burn=8000, steps=500, batch=False, fitter='MCMC', reload_problem=True, resume=False,
                  alpha=0.01):
        """
        Runs fit for Bumps object.

        :param burn: number of burn steps of the MCMC
        :param steps: number of production steps of the MCMC or the LM
        :param batch: uses the bumps 'batch' option, which silences most output and plot generation
        :param fitter: Default is 'MCMC', but 'LM' is supported, as well.
        :param reload_problem: determines whether the problem is reloaded from disk or whether the internally stored
                               problem is used, including any potential best-fit parameters from a previous run or
                               restore.
        :param resume: if True, resumes a fit from the store directory
        :param alpha: Dream convergence criterion
        :return: no return value
        """

        # Calling bumps functions directl
        model_file = os.path.join(self.spath, self.runfile) + '.py'
        mcmcpath = os.path.join(self.spath, self.mcmcpath)

        if not os.path.isdir(mcmcpath):
            os.mkdir(mcmcpath)

        # save model file in output directory
        shutil.copy(model_file, mcmcpath)

        if reload_problem or self.problem is None:
            self.problem = self.fnRestoreFitProblem()

        mapper = MPMapper.start_mapper(self.problem, None, cpus=0)
        monitors = None if not batch else []

        if fitter == 'MCMC':
            driver = FitDriver(fitclass=DreamFit, mapper=mapper, problem=self.problem, init='lhs', steps=steps,
                               burn=burn, monitors=monitors, xtol=1e-6, ftol=1e-8, alpha=alpha)
        elif fitter == 'LM':
            driver = FitDriver(fitclass=MPFit, mapper=mapper, problem=self.problem, monitors=monitors,
                               steps=steps, xtol=1e-6, ftol=1e-8)
        else:
            driver = None
        if resume:
            x, fx = driver.fit(resume=os.path.join(mcmcpath, self.runfile))
        else:
            x, fx = driver.fit()

        # try to deal with matplotlib memory leaks
        matplotlib.interactive(False)

        # .err and .par files
        self.problem.output_path = os.path.join(mcmcpath, self.runfile)
        save_best(driver, self.problem, x)

        # try to deal with matplotlib cache issues by deleting the cache
        fig = plt.figure()
        plt.figure().clear()
        plt.cla()
        plt.clf()
        plt.close("all")

        # don't know what files
        if 'models' in dir(self.problem):
            for M in self.problem.models:
                M.fitness.save(os.path.join(mcmcpath, self.runfile))
                break
        else:
            self.problem.fitness.save(os.path.join(mcmcpath, self.runfile))

        # .mcmc and .point files
        driver.save(os.path.join(mcmcpath, self.runfile))

        if not batch:
            # stat table and yet other files
            driver.show()
            # plots and other files
            driver.plot(os.path.join(mcmcpath, self.runfile))

    def fnSaveMolgroups(self, problem):
        # saves bilayer and protein information from a bumps / refl1d problem object into a mol.dat file
        # sequentially using the methods provided in molgroups
        fp = open(self.spath + '/mol.dat', "w")
        z = numpy.linspace(0, problem.dimension * problem.stepsize, problem.dimension, endpoint=False)
        try:
            problem.extra[0].fnWriteGroup2File(fp, 'bilayer', z)
            problem.extra[1].fnWriteGroup2File(fp, 'protein', z)
        except:
            problem.extra.fnWriteGroup2File(fp, 'bilayer', z)
        fp.close()
        stdout.flush()

    def fnUpdateModelPars(self, diNewPars):
        liParameters = list(self.diParameters.keys())
        # sort by number of appereance in runfile
        liParameters = sorted(liParameters, key=lambda keyitem: self.diParameters[keyitem]['number'])
        for element in liParameters:
            if element not in list(diNewPars.keys()):
                print('Parameter ' + element + ' not specified.')
                # check failed -> exit method
                return
            # else:
                # print(element + ' ' + str(diNewPars[element]))

        p = [diNewPars[parameter] for parameter in liParameters]
        self.problem.setp(p)
        self.problem.model_update()


