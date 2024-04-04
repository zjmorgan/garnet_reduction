from mantid.simpleapi import (FindPeaksMD,
                              PredictPeaks,
                              PredictSatellitePeaks,
                              CentroidPeaksMD,
                              IntegratePeaksMD,
                              PeakIntensityVsRadius,
                              FilterPeaks,
                              SortPeaksWorkspace,
                              DeleteWorkspace,
                              DeleteTableRows,
                              ExtractSingleSpectrum,
                              CombinePeaksWorkspaces,
                              CreatePeaksWorkspace,
                              CopySample,
                              CloneWorkspace,
                              SaveNexus,
                              LoadNexus,
                              AddPeakHKL,
                              HasUB,
                              mtd)

from mantid.kernel import V3D
from mantid.dataobjects import PeakShapeEllipsoid

import numpy as np

refl_cond_dict = {'P': 'Primitive',
                  'I': 'Body centred',
                  'F': 'All-face centred',
                  'R': 'Primitive', # rhombohedral axes
                  'R(obv)': 'Rhombohderally centred, obverse', # hexagonal axes
                  'R(rev)': 'Rhombohderally centred, reverse', # hexagonal axes
                  'A': 'A-face centred',
                  'B': 'B-face centred',
                  'C': 'C-face centred'}

class PeaksModel:

    def __init__(self):

        self.edge_pixels = 0

    def find_peaks(self, md,
                         peaks,
                         max_d,
                         density=1000,
                         max_peaks=50):
        """
        Harvest strong peak locations from Q-sample into a peaks table.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        max_d : float
            Maxium d-spacing enforcing lower limit of peak spacing.
        density : int, optional
            Threshold density. The default is 1000.
        max_peaks : int, optional
            Maximum number of peaks to find. The default is 50.

        """

        FindPeaksMD(InputWorkspace=md,
                    PeakDistanceTreshhold=2*np.pi/max_d,
                    MaxPeaks=max_peaks,
                    PeakFindingStrategy='VolumeNormalization',
                    DensityThresholdFactor=density,
                    EdgePixels=self.edge_pixels,
                    OutputWorkspace=peaks)

    def centroid_peaks(self, md, peaks, peak_radius):
        """
        Re-center peak locations using centroid within given radius

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integration region radius.

        """

        CentroidPeaksMD(InputWorkspace=md,
                        PeakRadius=peak_radius,
                        PeaksWorkspace=peaks,
                        OutputWorkspace=peaks)

    def integrate_peaks(self, md,
                              peaks,
                              peak_radius,
                              background_inner_fact=1,
                              background_outer_fact=1.5,
                              method='sphere'):
        """
        Integrate peaks using spherical or ellipsoidal regions.
        Ellipsoid integration adapts itself to the peak distribution.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integration region radius.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        method : str, optional
            Integration method. The default is 'sphere'.

        """

        background_inner_radius = peak_radius*background_inner_fact
        background_outer_radius = peak_radius*background_outer_fact

        IntegratePeaksMD(InputWorkspace=md,
                         PeaksWorkspace=peaks,
                         PeakRadius=peak_radius,
                         BackgroundInnerRadius=background_inner_radius,
                         BackgroundOuterRadius=background_outer_radius,
                         Ellipsoid=True if method == 'ellipsoid' else False,
                         FixQAxis=False,
                         FixMajorAxisLength=False,
                         UseCentroid=True,
                         MaxIterations=3,
                         ReplaceIntensity=True,
                         IntegrateIfOnEdge=True,
                         AdaptiveQBackground=False,
                         MaskEdgeTubes=False,
                         OutputWorkspace=peaks)

    def intensity_vs_radius(self, md,
                                  peaks,
                                  peak_radius,
                                  background_inner_fact=1,
                                  background_outer_fact=1.5,
                                  steps=51):
        """
        Integrate peak intensity with radius varying from zero to cut off.

        Parameters
        ----------
        md : str
            Name of Q-sample.
        peaks : str
            Name of peaks table.
        peak_radius : float
            Integrat region radius cut off.
        background_inner_fact : float, optional
            Factor of peak radius for background shell. The default is 1.
        background_outer_fact : float, optional
            Factor of peak radius for background shell. The default is 1.5.
        steps : int, optional
            Number of integration steps. The default is 51.

        Returns
        -------

        radius : list
            Peak radius.
        sig_noise : list
            Peak signal/noise ratio at lowest threshold.
        intens : list
            Peak intensity.

        """

        PeakIntensityVsRadius(InputWorkspace=md,
                              PeaksWorkspace=peaks,
                              RadiusStart=0.0,
                              RadiusEnd=peak_radius,
                              NumSteps=steps,
                              BackgroundInnerFactor=background_inner_fact,
                              BackgroundOuterFactor=background_outer_fact,
                              OutputWorkspace=peaks+'_intens_vs_rad',
                              OutputWorkspace2=peaks+'_sig/noise_vs_rad')

        ExtractSingleSpectrum(InputWorkspace=peaks+'_sig/noise_vs_rad',
                              OutputWorkspace=peaks+'_sig/noise_vs_rad/lowest',
                              WorkspaceIndex=0)

        peak_radius = mtd[peaks+'_sig/noise_vs_rad/lowest'].extractX().ravel()
        sig_noise = mtd[peaks+'_sig/noise_vs_rad/lowest'].extractY().ravel()
        intens = mtd[peaks+'_intens_vs_rad'].extractY()

        return peak_radius, sig_noise, intens

    def get_max_d_spacing(self, ws):
        """
        Obtain the maximum d-spacing from the oriented lattice.

        Parameters
        ----------
        ws : str
            Workspace with UB defined on oriented lattice.

        Returns
        -------
        d_max : float
            Maximum d-spacing.

        """

        if HasUB(Workspace=ws):

            ol = mtd[ws].sample().getOrientedLattice()

            return max([ol.a(), ol.b(), ol.c()])

    def predict_peaks(self, ws, peaks, centering, d_min, lamda_min, lamda_max):
        """
        Predict peak Q-sample locations with UB and lattice centering.

        +--------+-----------------------+
        | Symbol | Reflection condition  |
        +========+=======================+
        | P      | None                  |
        +--------+-----------------------+
        | I      | :math:`h+k+l=2n`      |
        +--------+-----------------------+
        | F      | :math:`h,k,l` unmixed |
        +--------+-----------------------+
        | R      | None                  |
        +--------+-----------------------+
        | R(obv) | :math:`-h+k+l=3n`     |
        +--------+-----------------------+
        | R(rev) | :math:`h-k+l=3n`      |
        +--------+-----------------------+
        | A      | :math:`k+l=2n`        |
        +--------+-----------------------+
        | B      | :math:`l+h=2n`        |
        +--------+-----------------------+
        | C      | :math:`h+k=2n`        |
        +--------+-----------------------+

        Parameters
        ----------
        ws : str
            Name of workspace to predict peaks with UB.
        peaks : str
            Name of peaks table.
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.

        """

        d_max = self.get_max_d_spacing(ws)

        PredictPeaks(InputWorkspace=ws,
                     WavelengthMin=lamda_min,
                     WavelengthMax=lamda_max,
                     MinDSpacing=d_min,
                     MaxDSpacing=d_max*1.2,
                     ReflectionCondition=refl_cond_dict[centering],
                     RoundHKL=True,
                     EdgePixels=self.edge_pixels,
                     OutputWorkspace=peaks)

    def predict_modulated_peaks(self, peaks,
                                      d_min,
                                      lamda_min,
                                      lamda_max,
                                      mod_vec_1=[0,0,0],
                                      mod_vec_2=[0,0,0],
                                      mod_vec_3=[0,0,0],
                                      max_order=0,
                                      cross_terms=False):
        """


        Parameters
        ----------
        ws : str
            Name of workspace to predict peaks with UB.
        peaks : str
            Name of peaks table.
        centering : str
            Lattice centering that provides the reflection condition.
        d_min : float
            The lower d-spacing resolution to predict peaks.
        lamda_min, lamda_max : float
            The wavelength band over which to predict peaks.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        """

        d_max = self.get_max_d_spacing(peaks)

        sat_peaks = peaks+'_sat'

        PredictSatellitePeaks(Peaks=peaks,
                              SatellitePeaks=sat_peaks,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              SaveModulationInfo=True,
                              IncludeIntegerHKL=False,
                              IncludeAllPeaksInRange=True,
                              WavelengthMin=lamda_min,
                              WavelengthMax=lamda_max,
                              MinDSpacing=d_min,
                              MaxDSpacing=d_max*10)

        CombinePeaksWorkspaces(LHSWorkspace=peaks,
                               RHSWorkspace=sat_peaks,
                               OutputWorkspace=peaks)

        DeleteWorkspace(Workspace=sat_peaks)

    def sort_peaks_by_hkl(self, peaks):
        """
        Sort peaks table by descending hkl values.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        columns = ['l', 'k', 'h']

        for col in columns:

            SortPeaksWorkspace(InputWorkpace=peaks,
                               ColumnNameToSortBy=col,
                               SortAscending=False,
                               OutputWorkspace=peaks)

    def sort_peaks_by_d(self, peaks):
        """
        Sort peaks table by descending d-spacing.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        SortPeaksWorkspace(InputWorkpace=peaks,
                           ColumnNameToSortBy='DSpacing',
                           SortAscending=False,
                           OutputWorkspace=peaks)

    def remove_duplicate_peaks(self, peaks):
        """
        Omit duplicate peaks from different based on indexing.
        Table will be sorted.

        Parameters
        ----------
        peaks : str
            Name of peaks table.

        """

        self.sort_peaks_by_hkl(peaks)

        for no in range(mtd[peaks].getNumberPeaks()-1,0,-1):

            if (mtd[peaks].getPeak(no).getHKL()-\
                mtd[peaks].getPeak(no-1).getHKL()).norm2() == 0:

                DeleteTableRows(TableWorkspace=peaks, Rows=no)

    def renumber_runs_by_index(self, ws, peaks):
        """
        Re-label the runs by index based on goniometer setting.

        Parameters
        ----------
        ws : str
            Name of workspace with goniometer indexing.
        peaks : str
            Name of peaks table.

        """

        run = mtd[ws].getExperimentInfo(0).run()

        n_gon = run.getNumGoniometers()

        Rs = np.array([run.getGoniometer(i).getR() for i in range(n_gon)])

        for no in range(mtd[peaks].getNumberPeaks()):

            peak = mtd[peaks].getPeak(no)

            R = peak.getGoniometerMatrix()

            ind = np.isclose(Rs, R).all(axis=(1,2))
            i = -1 if not np.any(ind) else ind.tolist().index(True)

            peak.setRunNumber(i+1)

    def load_peaks(self, filename, peaks):
        """
        Load peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        peaks : str
            Name of peaks table.

        """

        LoadNexus(Filename=filename,
                  OutputWorkspace=peaks)

    def save_peaks(self, filename, peaks):
        """
        Save peaks file.

        Parameters
        ----------
        filename : str
            Name of peaks file with extension .nxs.
        peaks : str
            Name of peaks table.

        """

        SaveNexus(Filename=filename,
                  InputWorkspace=peaks)

    def combine_peaks(self, peaks, merge):
        """
        Merge two peaks workspaces into one.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.
        merge : str
            Name of peaks table to be accumulated.

        """

        if not mtd.doesExist(merge):

            CloneWorkspace(InputWorkspace=peaks,
                           OutputWorkspace=merge)

        else:

            CombinePeaksWorkspaces(LHSWorkspace=merge,
                                   RHSWorkspace=peaks,
                                   OutputWorkspace=merge)

    def delete_peaks(self, peaks):
        """
        Remove peaks.

        Parameters
        ----------
        peaks : str
            Name of peaks table to be added.

        """

        if mtd.doesExist(peaks):

            DeleteWorkspace(Workspace=peaks)

    def remove_weak_peaks(self, peaks, sig_noise=3):
        """
        Filter out weak peaks based on signal-to-noise ratio.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        sig_noise : float, optional
            Minimum signal-to-noise ratio. The default is 3.

        """

        FilterPeaks(InputWorkspace=peaks,
                    OutputWorkspace=peaks,
                    FilterVariable='Signal/Noise',
                    FilterValue=sig_noise,
                    Operator='>',
                    Criterion='!=',
                    BankName='None')

    def create_peaks(self, ws, peaks):
        """
        Create a new peaks table.

        ws : str
            Name of workspace.
        peaks : str
            Name of peaks table.

        """

        CreatePeaksWorkspace(InstrumentWorkspace=ws,
                             NumberOfPeaks=0,
                             OutputWorkspace=peaks)

        CopySample(InputWorkspace=ws,
                   OutputWorkspace=peaks,
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)

    def add_peak(self, peaks, hkl):
        """
        Add a peak to an existing table.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        hkl : list
            Miller index.

        Returns
        -------
        None.

        """

        AddPeakHKL(Workspace='peak', HKL=hkl)

    def set_goniometer(self, peaks, R):
        """
        Update the goniometer on the run.

        Parameters
        ----------
        peaks : str
            Name of peaks table.
        R : 2d-array
            Goniometer matrix.

        """
        
        mtd[peaks].run().getGoniometer().setR(R)

class PeakModel:

    def __init__(self, peaks):

        self.peaks = peaks

    def get_number_peaks(self):
        """
        Total number of peaks in the table.

        Returns
        -------
        n : int
            Number of peaks

        """

        return mtd[self.peaks].getNumberPeaks()

    def set_peak_intensity(self, no, intens, sig):
        """
        Update the peak intensity

        Parameters
        ----------
        no : int
            Peak index number.
        intens : float
            Intensity.
        sig : float
            Uncertainty.

        """

        mtd[self.peaks].getPeak(no).setIntensity(intens)
        mtd[self.peaks].getPeak(no).setSigmaIntensity(sig)

    def get_peak_shape(self, no):
        """
        Obtain the peak shape parameters.

        Parameters
        ----------
        no : int
            Peak index number.

        Returns
        -------
        c0, c1, c2 : float
            Peak center.
        r0, r1, r2 : float
            Principal radii.
        v0, v1, v2 : list
            Principal axis directions.

        """

        Q0, Q1, Q2 = mtd[self.peaks].getPeak(no).getQSampleFrame()

        shape = eval(mtd[self.peaks].getPeak(no).getPeakShape().toJSON())

        c0 = Q0+shape['translation0']
        c1 = Q1+shape['translation1']
        c2 = Q2+shape['translation2']
 
        v0 = [float(val) for val in shape['direction0'].split(' ')]
        v1 = [float(val) for val in shape['direction1'].split(' ')]
        v2 = [float(val) for val in shape['direction2'].split(' ')]

        r0 = shape['radius0']
        r1 = shape['radius1']
        r2 = shape['radius2']

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def set_peak_shape(self, no, c0, c1, c2, r0, r1, r2, v0, v1, v2):
        """
        Update the shape of the peak.

        Parameters
        ----------
        no : int
            Peak index number.
        c0, c1, c2 : float
            Peak center.
        r0, r1, r2 : float
            Principal radii.
        v0, v1, v2 : list
            Principal axis directions.

        """

        R = mtd[self.peaks].getPeak(no).getGoniometerMatrix()

        Q = np.array([c0, c1, c2])
        Qx, Qy, Qz = R @ Q

        radii = [0, 0, 0]
        print(Qx, Qy, Qz)

        if -4*np.pi*Qz/np.linalg.norm(Q)**2 > 0:

            mtd[self.peaks].getPeak(no).setQSampleFrame(V3D(c0, c1, c2))

            radii = [r0, r1, r2]

        shape = PeakShapeEllipsoid([V3D(*v0), V3D(*v1), V3D(*v2)],
                                   radii,
                                   radii,
                                   radii)

        mtd[self.peaks].getPeak(no).setPeakShape(shape)

class StatisticsModel:

    def __init__(self, peaks):

        self.peaks = peaks+'_stats'

        CloneWorkspace(InputWorkspace=peaks, 
                       OutputWorkspace=self.peaks)

    def set_scale(self, scale='auto'):

        if scale == 'auto':
            scale = 1
            if mtd[self.peaks].getNumberPeaks() > 1:
                I_max = max(mtd[self.peaks].column('Intens'))
                if I_max > 0:
                    scale = 1e4/I_max

        for peak in mtd[self.peaks]:
            peak.setIntensity(scale*peak.getIntensity())
            peak.setSigmaIntensity(scale*peak.getSigmaIntensity())
