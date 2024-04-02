from mantid.simpleapi import (FindPeaksMD,
                              PredictPeaks,
                              PredictSatellitePeaks,
                              CentroidPeaksMD,
                              IntegratePeaksMD,
                              PeakIntensityVsRadius,
                              CombinePeaksWorkspace,
                              FilterPeaks,
                              SortPeaksWorkspace,
                              DeleteWorkspace,
                              DeleteTableRows,
                              HasUB,
                              mtd)

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
                         RelaceIntensity=True,
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

    def predict_peaks(self, ws, peaks, centering, d_min):
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

        """

        d_max = self.get_max_d_spacing(ws)

        PredictPeaks(InputWorkspace=ws,
                     WavelengthMin=self.wl_min,
                     WavelengthMax=self.wl_max,
                     MinDSpacing=d_min,
                     MaxDSpacing=d_max*1.2,
                     ReflectionCondition=refl_cond_dict[centering],
                     RoundHKL=True,
                     EdgePixels=self.edge_pixels,
                     OutputWorkspace=peaks)

    def predict_modulated_peaks(self, peaks,
                                      d_min,
                                      mod_vec_1=[0,0,0],
                                      mod_vec_2=[0,0,0],
                                      mod_vec_3=[0,0,0],
                                      max_order=0,
                                      cross_terms=False):

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
                              WavelengthMin=self.wl_min,
                              WavelengthMax=self.wl_max,
                              MinDSpacing=d_min,
                              MaxDSpacing=d_max*10)

        CombinePeaksWorkspace(LHSWorkspace=peaks,
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


