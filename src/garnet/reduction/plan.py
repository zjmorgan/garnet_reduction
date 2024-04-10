import os
import json

from garnet.config.instruments import beamlines

class ReductionPlan:

    def __init__(self):

        self.plan = None

    def validate_plan(self):

        assert self.plan['Instrument'] in beamlines.keys()

        if self.plan.get('UBFile') is not None:
            UB = self.plan['UBFile']
            assert os.path.exists(UB)
            assert os.path.splitext(UB)[1] == '.mat'

    def set_output(self, filename):
        """
        Change the output directory and name.

        Parameters
        ----------
        filename : str
            JSON file of reduction plan.

        """

        path = os.path.dirname(os.path.abspath(filename))
        name = os.path.splitext(os.path.basename(filename))[0]

        self.plan['OutputPath'] = path
        self.plan['OutputName'] = name

    def load_plan(self, filename):
        """
        Load a data reduction plan.

        Parameters
        ----------
        filename : str
            JSON file of reduction plan.

        """

        with open(filename, 'r') as f:

            self.plan = json.load(f)

        self.validate_plan()

        self.set_output(filename)
        runs = self.plan['Runs']
        if type(runs) is str:
            self.plan['Runs'] = self.runs_string_to_list(runs)

    def save_plan(self, filename):
        """
        Save a data reduction plan.

        Parameters
        ----------
        filename : str
            JSON file of reduction plan.

        """

        if self.plan is not None:

            self.set_output(filename)
            runs = self.plan['Runs']
            if type(runs) is list:
                self.plan['Runs'] = self.runs_list_to_string(runs)

            with open(filename, 'w') as f:

                json.dump(self.plan, f, indent=4)

    def runs_string_to_list(self, runs_str):
        """
        Convert runs string to list.

        Parameters
        ----------
        runs_str : str
            Condensed notation for run numbers.

        Returns
        -------
        runs : list
            Integer run numbers.

        """

        ranges = runs_str.split(',')
        runs = []
        for part in ranges:
            if ':' in part:
                start, end = map(int, part.split(':'))
                runs.extend(range(start, end + 1))
            else:
                runs.append(int(part))
        return runs

    def runs_list_to_string(self, runs):
        """
        Convert runs list to string.

        Parameters
        ----------
        runs : list
            Integer run numbers.

        Returns
        -------
        runs_str : str
            Condensed notation for run numbers.

        """

        if not runs:
            return ''

        runs.sort()
        result = []
        range_start = runs[0]

        for i in range(1, len(runs)):
            if runs[i] != runs[i-1] + 1:
                if range_start == runs[i-1]:
                    result.append(str(range_start))
                else:
                    result.append('{}:{}'.format(range_start, runs[i-1]))
                range_start = runs[i]

        if range_start == runs[-1]:
            result.append(str(range_start))
        else:
            result.append('{}:{}'.format(range_start, runs[-1]))

        run_str = ','.join(result)

        return run_str

    def generate_plan(self, instrument):
        """
        Create a template plan.

        Parameters
        ----------
        instrument : str
            Beamline name.

        """

        plan = {}

        assert instrument in beamlines.keys()
        params = beamlines[instrument]

        plan[instrument] = instrument
        plan['IPTS'] = 0
        plan['Runs'] = '1:2'
        if instrument == 'DEMAND':
            plan['Experiment'] = 1

        plan['UBFile'] = ''
        plan['Vanadium'] = ''

        if params['Facility'] == 'SNS':
            plan['FluxFile'] = ''
            plan['MaskFile'] = None
            plan['DetectorCalibration'] = None

        if instrument == 'CORELLI':
            plan['TubeCalibration'] = '/SNS/CORELLI/shared/calibration/tube'\
                                    + '/calibration_corelli_20200109.nxs.h5'
            plan['Elastic'] = False

        self.plan = plan

    def template_integration(self, instrument):
        """
        Generate template integration plan.

        Parameters
        ----------
        instrument : str
            Beamline name.

        Returns
        -------
        params : dict
            Integration plan.

        """

        params = {}
        params['Cell'] = 'Triclinic'
        params['Centering'] = 'P'
        params['ModVec1'] = [0,0,0]
        params['ModVec2'] = [0,0,0]
        params['ModVec3'] = [0,0,0]
        params['MaxOrder'] = 1
        params['MinD'] = 0.7
        params['Radius'] = 0.25

        return params

#     'Normalization': {
#         'Symmetry' : 'm-3m',
#         'Projections': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#         'Extents': [[-10, 10], [-10, 10], [-10, 10]],
#         'Bins': [201, 201, 201],
#     },
