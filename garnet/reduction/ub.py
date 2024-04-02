from mantid.simpleapi import (SelectCellWithForm,
                              ShowPossibleCells,
                              TransformHKL,
                              CalculatePeaksHKL,
                              IndexPeaks,
                              FindUBUsingFFT,
                              FindUBUsingLatticeParameters,
                              FindUBUsingIndexedPeaks,                              
                              OptimizeLatticeForCellType,
                              CalculateUMatrix,
                              HasUB,
                              LoadIsawUB,
                              SaveIsawUB,
                              CopySample,
                              mtd)

from mantid.geometry import PointGroupFactory

from mantid.utils.logging  import capture_logs

import numpy as np

lattice_group = {'Triclinic': '-1',
                 'Monoclinic': '2/m',
                 'Orthorhombic': 'mmm',
                 'Tetragonal': '4/mmm',
                 'Rhombohedral': '-3m',
                 'Hexagonal': '6/mmm',
                 'Cubic': 'm-3m'}

class UBModel():

    def __init__(self, peaks):
        """
        Tools for working with peaks and UB.

        Parameters
        ----------
        peaks : str
            Table of peaks.

        """

        self.peaks = mtd[peaks]

    def has_UB(self):
        """
        Check if peaks table has a UB determined.

        """

        return HasUB(Workspace=self.peaks)

    def save_UB(self, filename):
        """
        Save UB to file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        SaveIsawUB(InputWorkspace=self.peaks, Filename=filename)

    def load_UB(self, filename):
        """
        Load UB from file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """
        
        LoadIsawUB(InputWorkspace=self.peaks, Filename=filename)

    def determine_UB_with_niggli_cell(self, min_d, max_d, tol=0.1):
        """
        Determine UB with primitive lattice using min/max lattice constant.

        Parameters
        ----------
        min_d : float
            Minimum lattice parameter in ansgroms.
        max_d : float
            Maximum lattice parameter in angstroms.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingFFT(PeaksWorkspace=self.peaks, 
                       MinD=min_d,
                       MaxD=max_d,
                       Tolerance=tol)

    def determine_UB_with_lattice_parameters(self, a, 
                                                   b,
                                                   c,
                                                   alpha,
                                                   beta,
                                                   gamma,
                                                   tol=0.1):
        """
        Determine UB with prior known lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in ansgroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingLatticeParameters(PeaksWorkspace=self.peaks, 
                                     a=a,
                                     b=b,
                                     c=c,
                                     alpha=alpha,
                                     beta=beta,
                                     gamma=gamma,
                                     Tolerance=tol)

    def refine_UB_without_constraints(self, tol=0.1, sat_tol=None):
        """
        Refine UB with unconstrained lattice parameters.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        FindUBUsingIndexedPeaks(PeaksWorkspace=self.peaks,
                                Tolerance=tol,
                                ToleranceForSatellite=tol_for_sat)

    def refine_UB_with_constraints(self, cell, tol=0.1):
        """
        Refine UB with constraints corresponding to lattice system.
        
        +--------------+
        | Cubic        |
        +--------------+
        | Hexagonal    |
        +--------------+
        | Rhombohedral |
        +--------------+
        | Tetragonal   |
        +--------------+
        | Orthorhombic |
        +--------------+
        | Monoclinic   |
        +--------------+
        | Triclinic    |
        +--------------+

        Parameters
        ----------
        cell : float
            Lattice system.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        OptimizeLatticeForCellType(PeaksWorkspace=self.peaks,
                                   CellType=cell,
                                   Apply=True,
                                   Tolerance=tol)

    def refine_U_only(self, a, b, c, alpha, beta, gamma):
        """
        Refine the U orientation only.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in ansgroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        CalculateUMatrix(PeaksWorkspace=self.peaks, 
                         a=a,
                         b=b,
                         c=c,
                         alpha=alpha,
                         beta=beta,
                         gamma=gamma)

    def select_cell(self, number, tol=0.1):
        """
        Transform to conventional cell using form number.

        Parameters
        ----------
        number : int
            Form number.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """
        
        SelectCellWithForm(PeaksWorkspace=self.peaks,
                           FormNumber=number,
                           Apply=True,
                           Tolerance=tol)

    def possible_convetional_cells(self, max_error=0.2, permutations=True):
        """
        List possible conventional cells.

        Parameters
        ----------
        max_error : float, optional
            Max scalar error to report form numbers. The default is 0.2.
        permutations : bool, optional
            Allow permutations of the lattice. The default is True.

        Returns
        -------
        vals : list
            List of form numbers.

        """

        with capture_logs(level='notice') as logs:

            ShowPossibleCells(PeaksWorkspace=self.peaks,
                              MaxScalarError=max_error,
                              AllowPermuations=permutations,
                              BestOnly=False)

            vals = logs.getvalue()
            vals = [val for val in vals.split('\n') if val.startswith('Form')]

            return vals

    def transform_lattice(self, transform, tol=0.1):
        """
        Apply a cell transformation to the lattice.        

        Parameters
        ----------
        transform : 3x3 array-like
            Transform to apply to hkl values.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        hkl_trans = ','.join(['{},{},{}'.format(*row) for row in transform])
        
        TransformHKL(PeaksWorkspace=self.peaks,
                     Tolerance=tol,
                     HKLTransform=hkl_trans)

    def generate_lattice_transforms(self, cell):
        """
        Obtain poss

        Parameters
        ----------
        cell : str
            Latttice system.

        Returns
        -------
        transforms : dict
            Transform dictionary with symmetry operation as key.

        """

        symbol = lattice_group[cell]        

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transform = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = '{}: '.format(symop.getOrder())+symop.getIdentifier()
                transform[name] = T.tolist()

        return {key:transform[key] for key in sorted(transform.keys())}

    def index_peaks(self, tol=0.1,
                          sat_tol=None,
                          mod_vec_1=[0,0,0],
                          mod_vec_2=[0,0,0],
                          mod_vec_3=[0,0,0],
                          max_order=0,
                          cross_terms=False):
        """
        Index the peaks and calculate the lattice parameter uncertainties.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for sattelites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        Returns
        -------
        indexing : list
            Result of indexing including number indexed and errors.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        indexing = IndexPeaks(PeaksWorkspace=self.peaks,
                              Tolerance=tol,
                              ToleranceForSatellite=tol_for_sat,
                              RoundHKLs=True,
                              CommonUBForAll=True,
                              ModVector1=mod_vec_1,
                              ModVector2=mod_vec_2,
                              ModVector3=mod_vec_3,
                              MaxOrder=max_order,
                              CrossTerms=cross_terms,
                              SaveModulationInfo=True)

        return indexing

    def calculate_hkl(self):
        """
        Calculate hkl values without rounding.

        """
    
        CalculatePeaksHKL(PeaksWorkspace=self.peaks,
                          OverWrite=True)
    
    def copy_UB(self, workspace):
        """
        Copy UB to another workspace.

        Parameters
        ----------
        workspace : float
            Target workspace to copy the UB to.

        """

        CopySample(InputWorkspace=self.peak,
                   OutputWorkspace=workspace,
                   CopyName=False,
                   CopyMaterial=False,
                   CopyEnvironment=False,
                   CopyShape=False)