import os

import numpy as np

from garnet.reduction.data import DataModel
from garnet.reduction.crystallography import space_point, point_laue
from garnet.config.instruments import beamlines

class Normalization:

    def __init__(self, plan):

        self.plan = plan
        self.params = plan['Normalization']

        self.validate_params()

    def validate_params(self):

        symmetry = list(space_point.keys())+list(point_laue.keys())

        if self.params.get('Symmetry') is not None:

            assert self.params['Symmetry'] in symmetry

        assert len(self.params['Projections']) == 3
        assert np.linalg.det(self.params['Projections']) > 0

        assert len(self.params['Bins']) == 3
        assert (np.array(self.params['Bins']) > 0).all()
        assert np.product(self.params['Bins']) < 1001**3 # memory usage limit

        assert len(self.params['Extents']) == 3
        assert (np.diff(self.params['Extents'], axis=1) >= 0).all()

    def normalize(self):

        output_file = os.path.join(self.plan['OutputPath'],
                                   'normalization',
                                   self.plan['OutputName']+'.nxs')

        data = DataModel(beamlines[self.plan['Instrument']])

        data.load_generate_normalization(self.plan['VanadiumFile'],
                                         self.plan['FluxFile'])

        lamda_min, lamda_max = data.wavelength_band

        for run in self.plan['Runs']:

            data.load_data('data', self.plan['IPTS'], run)

            data.apply_calibration('data',
                                   self.plan.get('DetectorCalibration'),
                                   self.plan.get('TubeCalibration'))

            data.crop_for_normalization('data')

            data.convert_to_Q_sample('data', 'md', lorentz_corr=False)

            data.load_clear_UB(self.plan['UBFile'], 'data')
            
            #data.combine_histograms('md', 'combine')

        #peaks.save_peaks(output_file, 'combine')