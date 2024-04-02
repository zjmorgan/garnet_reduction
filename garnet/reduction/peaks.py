from mantid.simpleapi import (SetGoniometer,
                              FindPeaksMD,
                              PredictPeaks,
                              PredictSatellitePeaks,
                              CentroidPeaksMD,
                              IntegratePeaksMD,
                              AddPeakHKL,
                              PeakIntensityVsRadius,
                              CombinePeaksWorkspace,
                              FilterPeaks,
                              SortPeaksWorkspace,
                              DeleteWorkspace,
                              mtd)

import numpy as np

refl_cond_dict = {'P': 'Primitive', 
                  'I': 'Body centred',
                  'F': 'All-face centred',
                  'R': 'Rhombohderally centred, obverse',
                  'R(obv)': 'Rhombohderally centred, obverse',
                  'R(rev)': 'Rhombohderally centred, reverse',
                  'A': 'A-face centred',
                  'B': 'B-face centred',
                  'C': 'C-face centred'}

class PeaksModel:

    def __init__(self):

        self.edge_pixels = 0

    def find_peaks(self, md, 
                         peaks,
                         max_d=15,
                         density=1000,
                         max_peaks=50):

        FindPeaksMD(InputWorkspace=md,
                    PeakDistanceTreshhold=2*np.pi/max_d,
                    MaxPeaks=max_peaks,
                    PeakFindingStrategy='VolumeNormalization',
                    DensityThresholdFactor=density,
                    EdgePixels=self.edge_pixels,
                    OutputWorkspace=peaks)

    def centroid_peaks(self, md, peaks, peak_radius):

        CentroidPeaksMD(InputWorkspace=md,
                        PeakRadius=peak_radius,
                        PeaksWorkspace=peaks,
                        OutputWorkspace=peaks)

    def integrate_peaks(self, md,
                              peaks,
                              peak_radius,
                              background_inner_fact=0,
                              background_outer_fact=0,
                              method='sphere'):

        background_inner_radius = peak_radius*background_inner_fact
        background_outer_radius = peak_radius*background_outer_fact

        IntegratePeaksMD(InputWorkspace=md,
                         PeaksWorkspace=peaks,
                         PeakRadius=peak_radius,
                         BackgroundInnerRadius=background_inner_radius,
                         BackgroundOuterRadius=background_outer_radius,
                         Ellipsoid=True if method == 'ellipsoid' else False,
                         FixQAxis=True,
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
                                  background_inner_fact=0,
                                  background_outer_fact=0,
                                  steps=51):

        PeakIntensityVsRadius(InputWorkspace=md,
                              PeaksWorkspace=peaks,
                              RadiusStart=0.0, 
                              RadiusEnd=peak_radius,
                              NumSteps=steps,
                              BackgroundInnerFactor=background_inner_fact,
                              BackgroundOuterFactor=background_outer_fact,
                              OutputWorkspace='peak_vs_rad')

    def get_max_d_spacing(self, ws):

        ol = mtd[ws].sample().getOrientedLattice()       

        return max([ol.a(), ol.b(), ol.c()])

    def predict_peaks(self, ws, refl_cond, d_min):

        d_max = self.get_max_d_spacing(ws)

        PredictPeaks(InputWorkspace=ws,
                     WavelengthMin=self.wl_min,
                     WavelengthMax=self.wl_max,
                     MinDSpacing=d_min,
                     MaxDSpacing=d_max,
                     ReflectionCondition=refl_cond_dict[refl_cond],
                     RoundHKL=True,
                     EdgePixels=self.edge_pixels,
                     OutputWorkspace='predict')

    def predict_satellite_peaks(self, peaks,
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
                              WavelengthMin=self.wl_min,
                              WavelengthMax=self.wl_max,
                              MinDSpacing=d_min,
                              MaxDSpacing=d_max)

        CombinePeaksWorkspace(LHSWorkspace=peaks,
                              RHSWorkspace=sat_peaks,
                              OutputWorkspace=peaks)

        DeleteWorkspace(Workspace=sat_peaks)

    def sort_peaks_by_d_hkl(self, peaks):
        """
        Sort peaks table by d-spacing and hkl values.

        Parameters
        ----------
        peaks : float
            Name of peaks table.

        """

        columns = ['l', 'k', 'h', 'DSpacing']

        for col in columns:

            SortPeaksWorkspace(InputWorkpace=peaks,
                               ColumnNameToSortBy=col,
                               SortAscending=False,
                               OutputWorkspace=peaks)