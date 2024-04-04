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
                              ReplicateMD,
                              BinMD,
                              DivideMD,
                              MDNorm,
                              ConvertUnits,
                              CropWorkspaceForMDNorm,
                              ClearUB,
                              LoadIsawUB,
                              mtd)

def DataModel(instrument_config):

    if type(instrument_config['Wavelength']) is list:
        return LaueData(instrument_config)
    else:
        return MonochromaticData(instrument_config)

class BaseDataModel:

    def __init__(self, instrument_config):

        self.instrument_config = instrument_config
        self.instrument = self.instrument_config['FancyName']

        facility = self.instrument_config['Facility']
        name = self.instrument_config['Name']
        iptspath = 'IPTS-{}'
        rawfile = self.instrument_config['RawFile']

        raw_file_path = os.path.join('/', facility, name, iptspath, rawfile)

        self.raw_file_path = raw_file_path

        self.gon_axis = 6*[None]
        gon_axis_names = self.instrument_config.get('GoniometerAxisNames')
        gon = self.instrument_config.get('Goniometer')
        if gon_axis_names is None:
            gon_axis_names = list(gon.keys())
            
        gon_ind = 0
        for i, name in enumerate(gon.keys()):
            axis = gon[name]
            if gon_axis_names is not None:
                name = gon_axis_names[i]
                self.gon_axis[gon_ind] = ','.join(5*['{}']).format(name, *axis)
                gon_ind += 1

        wl = instrument_config['Wavelength']

        self.wavelength_band = wl if type(wl) is list else [0.98*wl, 1.02*wl]
        self.lamda = np.mean(wl) if type(wl) is list else wl

        self.k_min = 2*np.pi/np.max(self.wavelength_band)
        self.k_max = 2*np.pi/np.min(self.wavelength_band)

    def load_clear_UB(self, filename, ws):
        """
        Load UB from file and replace.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.
        ws : str, optional
           Name of data.

        """

        ClearUB(Workspace=ws)
        LoadIsawUB(InputWorkspace=ws, Filename=filename)

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

        if type(runs) is int:
            runs = [runs]

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

    def set_goniometer(self, ws):
        """
        Set the goniomter motor angles

        Parameters
        ----------
        ws : str, optional
           Name of raw data.

        """

        SetGoniometer(Workspace=ws,
                      Goniometers='None, Specify Individually',
                      Axis0=self.gon_axis[0],
                      Axis1=self.gon_axis[1],
                      Axis2=self.gon_axis[2],
                      Axis3=self.gon_axis[3],
                      Axis4=self.gon_axis[4],
                      Axis5=self.gon_axis[5],
                      Average=self.laue)

    def calulate_binning_from_bins(self, xmin, xmax, bins):
        """
        Determine the binning from the number of bins.

        Parameters
        ----------
        xmin : float
            Minimum bin center.
        xmax : float
            Maximum bin center.
        bins : TYPE
            Number of bins.

        Returns
        -------
        min_edge : float
            Minimum bin edge.
        max_edge : float
            Maximum bin edge.
        step : float
            Bin step.

        """

        step = (xmax-xmin)/(bins-1)

        min_bin = xmin-0.5*step
        max_bin = xmax+0.5*step

        return min_bin, max_bin, step

    def calulate_binning_from_step(xmin, xmax, step):
        """
        Determine the binning from step size.

        Parameters
        ----------
        xmin : float
            Minimum bin center.
        xmax : float
            Maximum bin center.
        step : float
            Bin step.

        Returns
        -------
        min_edge : float
            Minimum bin edge.
        max_edge : float
            Maximum bin edge.
        bins : int
            Number of bins.

        """

        bins = np.ceil((xmax-xmin)/step) + 1

        min_bin = xmin-0.5*step
        max_bin = xmax+0.5*step

        return min_bin, max_bin, bins

    def extract_bin_info(self, ws):
        """
        Obtain the bin information from a histogram.

        Parameters
        ----------
        ws : str
            Name of histogram.

        Returns
        -------
        signal : array
            Data signal.
        error : array
            Data uncertanies.
        x0, x1, ... : array
            Bin center coordinates.

        """
        
        signal = mtd[ws].getSignalArray().copy()
        error = np.sqrt(mtd[ws].getErrorSquaredArray())

        dims = [mtd[ws].getDimension(i) for i in range(mtd[ws].getNumDims())]

        xs = [np.linspace(dim.getMinimum(),
                          dim.getMaximum(),
                          dim.getNBoundaries()) for dim in dims]

        xs = [0.5*(x[1:]+x[:-1]) for x in xs]

        xs = np.meshgrid(*xs, indexing='ij')
        
        return signal, error, *xs

class MonochromaticData(BaseDataModel):

    def __init__(self, instrument_config):

        super(MonochromaticData).__init__(instrument_config)

        self.laue = False

    def load_data(self, histo_name, IPTS, runs):
        """
        Load raw data into detector counts vs rotation index.

        Parameters
        ----------
        histo_name : str
            Name of raw histogram data.
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
                                 OutputWorkspace=histo_name)
            ei = mtd[histo_name].getExperimentInfo(0)
            si = ei.spectrumInfo()
            n_det = ei.getInstrument().getNumberDetectors()
            self.theta_max = 0.5*np.max([si.twoTheta(i) for i in range(n_det)])
            self.scale = ei.run().getProperty('time').value

        else:
            LoadWANDSCD(Filename=filenames,
                        Grouping='None',
                        OutputWorkspace=histo_name)
            run = mtd[histo_name].getExperimentInfo(0).run()
            self.theta_max = 0.5*np.max(run.getProperty('TwoTheta').value)
            self.scale = run.getProperty('duration').value

        self.Q_max = 4*np.pi/self.lamda*np.sin(0.5*self.theta_max)

        self.set_goniometer(histo_name)

    def convert_to_Q_sample(self, histo_name, md_name, lorentz_corr=False):
        """
        Convert raw data to Q-sample.

        Parameters
        ----------
        histo_name : str
            Name of raw histogram data.
        md_name : str
            Name of Q-sample workspace.
        lorentz_corr : bool, optional
            Apply Lorentz correction. The default is False.

        """

        if mtd.doesExist(histo_name):

            Q_min_vals, Q_max_vals = self.get_min_max_values()

            ConvertHFIRSCDtoMDE(InputWorkspace=histo_name,
                                Wavelength=self.wavelength,
                                LorentzCorrection=lorentz_corr,
                                MinValues=Q_min_vals,
                                MaxValues=Q_max_vals,
                                OutputWorkspace=md_name)

    def load_generate_normalization(self, filename, histo_name=None):
        """
        Load a vanadium file and generate normalization data.
        Provided a histogram workspace name, generate corresponding shape.

        Parameters
        ----------
        filename : str
            Vanadium file.
        histo_name : str, optional
            Name of raw histogram data.

        """

        if not mtd.doesExist('van'):

            if self.instrument == 'DEMAND':
                HB3AAdjustSampleNorm(Filename=filename,
                                     OutputType='Detector',
                                     NormaliseBy='None',
                                     OutputWorkspace='van')

            else:
                LoadWANDSCD(Filename=filename,
                            Grouping='None',
                            OutputWorkspace='van')

            if histo_name is not None:

                if mtd.doesExist(histo_name):

                    ws_name = '{}_van'.format(histo_name)

                    ReplicateMD(ShapeWorkspace=histo_name,
                                DataWorkspace='van',
                                OutputWorkspace=ws_name)

                    self.set_goniometer(ws_name)

                    signal = mtd[ws_name].getSignalArray().copy()
                    mtd[ws_name].setSignalArray(signal*self.scale)

                    Q_min_vals, Q_max_vals = self.get_min_max_values()

                    ConvertHFIRSCDtoMDE(InputWorkspace=ws_name,
                                        Wavelength=self.wavelength,
                                        LorentzCorrection=False,
                                        MinValues=Q_min_vals,
                                        MaxValues=Q_max_vals,
                                        OutputWorkspace='norm')

    def normalize_to_Q_sample(self, md, extents, bins):
        """
        Histogram data into normalized Q-sample.

        Parameters
        ----------
        md : str
            3D Q-sample data.
        extents : list
            Min/max pairs for each dimension.
        bins : list
            Number of bins for each dimension.

        """

        if mtd.doesExist(md) and mtd.doesExist('norm'):

            BinMD(InputWorkspace=md,
                  AxisAligned=False,
                  BasisVector0='Q_sample_x,Angstrom^-1,1.0,0.0,0.0',
                  BasisVector1='Q_sample_y,Angstrom^-1,0.0,1.0,0.0',
                  BasisVector2='Q_sample_z,Angstrom^-1,0.0,0.0,1.0',
                  OutputExtents=extents,
                  OutputBins=bins,
                  OutputWorkspace=md+'_data')

            BinMD(InputWorkspace='norm',
                  AxisAligned=False,
                  BasisVector0='Q_sample_x,Angstrom^-1,1.0,0.0,0.0',
                  BasisVector1='Q_sample_y,Angstrom^-1,0.0,1.0,0.0',
                  BasisVector2='Q_sample_z,Angstrom^-1,0.0,0.0,1.0',
                  OutputExtents=extents,
                  OutputBins=bins,
                  OutputWorkspace=md+'_norm')

            DivideMD(LHSWorkspace=md+'_data',
                     RHSWorkspace=md+'_norm',
                     OutputWorkspace=md+'_result')

            data, _, Q0, Q1, Q2 = self.extract_bin_info(md+'_data')
            norm, _, Q0, Q1, Q2 = self.extract_bin_info(md+'_norm')

            return data, norm, Q0, Q1, Q2

class LaueData(BaseDataModel):

    def __init__(self, instrument_config):

        super(LaueData, self).__init__(instrument_config)

        self.laue = True

    def load_data(self, event_name, IPTS, runs):
        """
        Load raw data into time-of-flight vs counts.

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.
        IPTS : int
            Proposal number.
        runs : list, int
            List of run number(s).

        """

        filenames = self.file_names(IPTS, runs)

        Load(Filename=filenames,
             OutputWorkspace=event_name)

        if not mtd.doesExist('detectors'):

            PreprocessDetectorsToMD(InputWorkspace=event_name,
                                    OutputWorkspace='detectors')

            two_theta = mtd['detectors'].column('TwoTheta')
            self.theta_max = 0.5*np.max(two_theta)

        self.calculate_maximum_Q()

        self.set_goniometer(event_name)
    
    def calculate_maximum_Q(self):
        """
        Update maximum Q.

        """

        lamda_min = np.min(self.wavelength_band)
        self.Q_max = 4*np.pi/lamda_min*np.sin(self.theta_max)

    def apply_calibration(self, event_name, 
                                detector_calibration,
                                tube_calibration=None):
        """
        Apply detector calibration.

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.
        detector_calibration : str
            Detector calibration as either .xml or .DetCal.
        tube_calibration : str, optional
            CORELLI-only tube calibration. The default is None.

        """

        if tube_calibration is not None:

            if not mtd.doesExist('tube_table'):

                LoadNexus(Filename=tube_calibration,
                          OutputWorkspace='tube_table')

            ApplyCalibration(Workspace=event_name,
                             CalibrationTable='tube_table')

        if detector_calibration is not None:

            if os.path.splitext(detector_calibration)[1] == '.xml':

                LoadParameterFile(Workspace=event_name,
                                  Filename=detector_calibration)

            else:

                LoadIsawDetCal(InputWorkspace=event_name,
                               Filename=detector_calibration)

    def convert_to_Q_sample(self, event_name, md_name, lorentz_corr=False):
        """
        Convert raw data to Q-sample.

        Parameters
        ----------
        event_name : str
            Name of raw event_name data.
        md_name : str
            Name of Q-sample workspace.
        lorentz_corr : bool, optional
            Apply Lorentz correction. The default is False.

        """

        if mtd.doesExist(event_name):

            Q_min_vals, Q_max_vals = self.get_min_max_values()

            ConvertToMD(InputWorkspace=event_name,
                        QDimensions='Q3D',
                        dEAnalysisMode='Elastic',
                        Q3DFrames='Q_sample',
                        LorentzCorrection=lorentz_corr,
                        MinValues=Q_min_vals,
                        MaxValues=Q_max_vals,
                        OutputWorkspace=md_name)

    def load_generate_normalization(self, vanadium_file, spectrum_file):
        """
        Load a vanadium file and generate normalization data.

        Parameters
        ----------
        vanadium_file : str
            Solid angle file.
        spectrum_file : str
            Flux file.

        """

        if not mtd.doesExist('sa'):
    
            LoadNexus(Filename=vanadium_file,
                      OutputWorkspace='sa')

        if not mtd.doesExist('flux'):

            LoadNexus(Filename=spectrum_file,
                      OutputWorkspace='flux')
        
            self.k_min = mtd['flux'].getXDimension().getMinimum()
            self.k_max = mtd['flux'].getXDimension().getMaximum()
            
            lamda_min = 2*np.pi/self.k_max
            lamda_max = 2*np.pi/self.k_min
            
            self.wavelength_band = [lamda_min, lamda_max]

    def crop_for_normalization(self, event_name):
        """
        Convert units to momentum and crop to wavelength band.

        event_name : str
            Name of raw event_name data.
        
        """

        if mtd.doesExist(event_name):

            ConvertUnits(InputWorkspace=event_name,
                         OutputWorkspace=event_name,
                         Target='Momentum')
    
            CropWorkspaceForMDNorm(InputWorkspace=event_name,
                                   XMin=self.k_min,
                                   XMax=self.k_max,
                                   OutputWorkspace=event_name)

    def normalize_to_Q_sample(self, md, extents, bins):
        """
        Histogram data into normalized Q-sample.

        Parameters
        ----------
        md : str
            3D Q-sample data.
        extents : list
            Min/max pairs for each dimension.
        bins : list
            Number of bins for each dimension.

        """

        if mtd.doesExist(md) and mtd.doesExist('sa') and mtd.doesExist('flux'):

            Q0_min, Q0_max, Q1_min, Q1_max, Q2_min, Q2_max = extents           

            nQ0, nQ1, nQ2 = bins
                
            Q0_min, Q0_max, dQ0 = self.calulate_binning_from_bins(Q0_min,
                                                                  Q0_max, nQ0)

            Q1_min, Q1_max, dQ1 = self.calulate_binning_from_bins(Q1_min,
                                                                  Q1_max, nQ1)

            Q2_min, Q2_max, dQ2 = self.calulate_binning_from_bins(Q2_min,
                                                                  Q2_max, nQ2)

            MDNorm(InputWorkspace=md,
                   RLU=False,
                   SolidAngleWorkspace='sa',
                   FluxWorkspace='flux',
                   Dimension0Binning='{},{},{}'.format(Q0_min,dQ0,Q0_max),
                   Dimension1Binning='{},{},{}'.format(Q1_min,dQ1,Q1_max),
                   Dimension2Binning='{},{},{}'.format(Q2_min,dQ2,Q2_max),
                   OutputWorkspace=md+'_result',
                   OutputDataWorkspace=md+'_data',
                   OutputNormalizationWorkspace=md+'_norm')

            data, _, Q0, Q1, Q2 = self.extract_bin_info(md+'_data')
            norm, _, Q0, Q1, Q2 = self.extract_bin_info(md+'_norm')

            return data, norm, Q0, Q1, Q2