import os

import numpy as np

from mantid.simpleapi import (Load,
                              LoadNexus,
                              LoadParameterFile,
                              LoadIsawDetCal,
                              ApplyCalibration,
                              PreprocessDetectorsToMD,
                              SetGoniometer,
                              LoadWANDSCD,
                              HB3AAdjustSampleNorm,
                              ConvertToMD,
                              ConvertHFIRSCDtoMDE,
                              mtd)

class DataModel:

    def __init__(self, instrument_config):

        self.instrument_config = instrument_config
        self.instrument = self.instrument_config['FancyName']

        facility = self.instrument_config['Facility']
        name = self.instrument_config['Name']
        iptspath = 'IPTS-{}'
        rawfile = self.instrument_config['RawFile']

        raw_file_path = os.path.join(facility, name, iptspath, rawfile)

        self.raw_file_path = raw_file_path

        self.gon_axis = 6*[None]
        gon_axis_names = self.instrument_config.get('GoniometerAxisNames')
        gon = self.instrument_config.get('Goniometer')

        gon_ind = 0
        for i, name in enumerate(gon.keys()):
            axis = gon[name]
            if gon_axis_names is not None:
                name = gon_axis_names[i]
                self.gon_axis[gon_ind] = ','.join(6*['{}']).format(name, *axis)
                gon_ind += 1

        wl = instrument_config['Wavelength']

        self.wavelength_band = wl if type(wl) is list else [0.98*wl, 1.02*wl]
        self.lamda = np.mean(wl) if type(wl) is list else wl

        if type(self.instrument_params['Wavelength']) is list:
            return LaueData()
        else:
            return MonochromaticData()

    def file_names(self, IPTS, runs):
        """
        Complete file paths.

        Parameters
        ----------
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).

        Returns
        -------
        filenames : str
            Comma separated filepaths.

        """

        filename = self.raw_file_path
        filenames = ','.join([filename.format(IPTS, run) for run in runs])
        return filenames

    def get_min_max_values(self):
        """
        The minimum and maximum Q-values.

        Returns
        -------
        Q_min_vals: list
            Minumum Q.
        Q_max_vals: list
            Maximum Q.

        """

        return  3*[-self.Q_max], 3*[+self.Q_max]

    def set_goniometer(self):
        """
        Set the goniomter motor angles

        """

        SetGoniometer(Workspace='data',
                      Goniometers='None, Specify Individually',
                      Axis0=self.gon_axis[0],
                      Axis1=self.gon_axis[1],
                      Axis2=self.gon_axis[2],
                      Axis3=self.gon_axis[3],
                      Axis4=self.gon_axis[4],
                      Axis5=self.gon_axis[5],
                      Average=self.laue)

class MonochromaticData(DataModel):

    def __init__(self):

        super(MonochromaticData, self).__init__()

        self.laue = False

    def load_data(self, IPTS, runs):
        """
        Load raw data into detector counts vs rotation index.

        Parameters
        ----------
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).

        """

        filenames = self.file_names(IPTS, runs)

        if self.instrument == 'DEMAND':
            HB3AAdjustSampleNorm(Filename=filenames,
                                 OutputType='Detector',
                                 NormaliseBy='None',
                                 OutputWorkspace='data')
            ei = mtd['data'].getExperimentInfo(0)
            si = ei.spectrumInfo()
            n_det = ei.getInstrument().getNumberDetectors()
            theta_max = 0.5*np.max([si.twoTheta(i) for i in range(n_det)])

        else:
            LoadWANDSCD(Filename=filenames,
                        Grouping='4x4',
                        OutputWorkspace='data')
            run = mtd['data'].getExperimentInfo(0).run()
            theta_max = 0.5*np.max(run.getProperty('TwoTheta').value)
        
        self.Q_max = 4*np.pi/self.lamda*np.sin(0.5*theta_max)

        self.set_goniometer()

    def convert_to_Q_sample(self, lorentz_correction=False):
        """
        Convert raw data to Q-sample.

        Parameters
        ----------
        lorentz_correction : bool, optional
            Apply Lorentz correction. The default is False.

        """

        if mtd.doesExist('data'):

            Q_min_vals, Q_max_vals = self.get_min_max_values()

            ConvertHFIRSCDtoMDE(InputWorkspace='data',
                                Wavelength=self.wavelength,
                                LorentzCorrection=lorentz_correction,
                                MinValues=Q_min_vals,
                                MaxValues=Q_max_vals,
                                OutputWorkspace='md')


class LaueData(DataModel):

    def __init__(self):

        super(LaueData, self).__init__()

        self.laue = True

    def load_data(self, IPTS, runs):
        """
        Load raw data into time-of-flight vs counts.

        Parameters
        ----------
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).

        """

        filenames = self.file_names(IPTS, runs)

        Load(Filename=filenames,
             OutputWorkspace='data')

        PreprocessDetectorsToMD(InputWorkspace='data',
                                OutputWorkspace='detectors')

        two_theta = mtd['detectors'].column('TwoTheta')
        lamda_min = np.min(self.wavelength_band)
        theta_max = 0.5*np.max(two_theta)
        self.Q_max = 4*np.pi/lamda_min*np.sin(theta_max)

        self.set_goniometer()

    def apply_calibration(self, detector_calibration, tube_calibration=None):
        """
        Apply detector calibration.

        Parameters
        ----------
        detector_calibration : str
            Detector calibration as either .xml or .DetCal.
        tube_calibration : str, optional
            CORELLI only tube calibration. The default is None.

        """

        if tube_calibration is not None:

            if not mtd.doesExist('tube_table'):

                LoadNexus(Filename=tube_calibration,
                          OutputWorkspace='tube_table')

            ApplyCalibration(Workspace='data',
                             CalibrationTable='tube_table')

        if detector_calibration is not None:

            if os.path.splitext(detector_calibration)[1] == '.xml':

                LoadParameterFile(Workspace='data',
                                  Filename=detector_calibration)

            else:

                LoadIsawDetCal(InputWorkspace='data',
                               Filename=detector_calibration)

    def convert_to_Q_sample(self, lorentz_correction=False):
        """
        Convert raw data to Q-sample.

        Parameters
        ----------
        lorentz_correction : bool, optional
            Apply Lorentz correction. The default is False.

        """

        if mtd.doesExist('data'):

            Q_min_vals, Q_max_vals = self.get_min_max_values()

            ConvertToMD(InputWorkspace='data',
                        QDimensions='Q3D',
                        dEAnalysisMode='Elastic',
                        Q3DFrames='Q_sample',
                        LorentzCorrection=lorentz_correction,
                        MinValues=Q_min_vals,
                        MaxValues=Q_max_vals,
                        PreprocDetectorsWS='detectors',
                        OutputWorkspace='md')
        
        