import os

import yaml

from garnet.config.instruments import beamlines


class Dumper(yaml.Dumper):
    def represent_list(self, data):
        return self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


Dumper.add_representer(list, Dumper.represent_list)


def save_YAML(output, filename):
    """Save reduction output file.

    Parameters
    ----------
    output : TYPE
        DESCRIPTION.
    filename : str
        Output file name.

    Returns
    -------
    None.

    """
    with open(filename, "w") as f:
        yaml.dump(output, f, Dumper=Dumper, sort_keys=False)


class ReductionPlan:
    def __init__(self):
        self.plan = None

    def validate_plan(self):
        assert self.plan["Instrument"] in beamlines.keys()

        if self.plan.get("UBFile") is not None:
            UB = self.plan["UBFile"]
            print("UB", UB)
            assert os.path.exists(UB)
            assert os.path.splitext(UB)[1] == ".mat"

        nxs_items = ["VanadiumFile", "FluxFile", "TubeCalibration", "BackgroundFile"]

        for item in nxs_items:
            if self.plan.get(item) is not None:
                fname = self.plan[item]
                assert os.path.exists(fname)
                assert os.path.splitext(fname)[1] in [".nxs", ".h5"]

        if self.plan.get("MaskFile") is not None:
            mask = self.plan["MaskFile"]
            assert os.path.exists(mask)
            assert os.path.splitext(mask)[1] == ".xml"

        if self.plan.get("DetectorCalibration") is not None:
            detcal = self.plan["DetectorCalibration"]
            assert os.path.exists(detcal)
            assert os.path.splitext(detcal)[1].lower() in [".xml", ".detcal"]

    def set_output(self, filename):
        """Change the output directory and name.

        Parameters
        ----------
        filename : str
            yaml file of reduction plan.

        """
        path = os.path.dirname(os.path.abspath(filename))
        name = os.path.splitext(os.path.basename(filename))[0]

        self.plan["OutputPath"] = path
        self.plan["OutputName"] = name

    def load_plan(self, filename):
        """Load a data reduction plan.

        Parameters
        ----------
        filename : str
            yaml file of reduction plan.

        """
        with open(filename, "r") as f:
            self.plan = yaml.safe_load(f)

        self.validate_plan()

        self.set_output(filename)
        runs = self.plan["Runs"]
        if isinstance(runs, str):
            self.plan["Runs"] = self.runs_string_to_list(runs)

    def save_plan(self, filename):
        """Save a data reduction plan.

        Parameters
        ----------
        filename : str
            yaml file of reduction plan.

        """
        if self.plan is not None:
            self.set_output(filename)
            runs = self.plan["Runs"]
            if isinstance(runs, list):
                self.plan["Runs"] = self.runs_list_to_string(runs)

            save_YAML(self.plan, filename)

    def runs_string_to_list(self, runs_str):
        """Convert runs string to list.

        Parameters
        ----------
        runs_str : str
            Condensed notation for run numbers.

        Returns
        -------
        runs : list
            Integer run numbers.

        """
        ranges = runs_str.split(",")
        runs = []
        for part in ranges:
            if ":" in part:
                start, end = map(int, part.split(":"))
                runs.extend(range(start, end + 1))
            else:
                runs.append(int(part))
        return runs

    def runs_list_to_string(self, runs):
        """Convert runs list to string.

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
            return ""

        runs.sort()
        result = []
        range_start = runs[0]

        for i in range(1, len(runs)):
            if runs[i] != runs[i - 1] + 1:
                if range_start == runs[i - 1]:
                    result.append(str(range_start))
                else:
                    result.append(f"{range_start}:{runs[i - 1]}")
                range_start = runs[i]

        if range_start == runs[-1]:
            result.append(str(range_start))
        else:
            result.append(f"{range_start}:{runs[-1]}")

        run_str = ",".join(result)

        return run_str

    def generate_plan(self, instrument):
        """Create a template plan.

        Parameters
        ----------
        instrument : str
            Beamline name.

        """
        plan = {}

        assert instrument in beamlines.keys()
        params = beamlines[instrument]

        plan["Instrument"] = instrument
        plan["IPTS"] = 0
        plan["Runs"] = "1:2"

        if instrument == "DEMAND":
            plan["Experiment"] = 1

        if params["Facility"] == "HFIR":
            plan["UBFile"] = None
        else:
            plan["UBFile"] = ""

        plan["VanadiumFile"] = ""
        plan["BackgroundFile"] = None

        if params["Facility"] == "SNS":
            plan["FluxFile"] = ""
            plan["MaskFile"] = None
            plan["DetectorCalibration"] = None

        if instrument == "CORELLI":
            plan["TubeCalibration"] = "/SNS/CORELLI/shared/calibration/tube" + "/calibration_corelli_20200109.nxs.h5"
            plan["Elastic"] = False
            plan["TimeOffset"] = None

        self.plan = plan

        self.plan["Integration"] = self.template_integration(instrument)
        self.plan["Normalization"] = self.template_normalization()

    def template_integration(self, instrument):
        """Generate template integration plan.

        Parameters
        ----------
        instrument : str
            Beamline name.

        Returns
        -------
        params : dict
            Integration plan.

        """
        inst_config = beamlines[instrument]

        wl = inst_config["Wavelength"]
        min_d = max(wl) / 2 if isinstance(wl, list) else wl / 2

        params = {}
        params["Cell"] = "Triclinic"
        params["Centering"] = "P"
        params["ModVec1"] = [0, 0, 0]
        params["ModVec2"] = [0, 0, 0]
        params["ModVec3"] = [0, 0, 0]
        params["MaxOrder"] = 0
        params["CrossTerms"] = False
        params["MinD"] = min_d
        params["Radius"] = 0.2

        return params

    def template_normalization(self):
        """Generate template integration plan.

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
        params["Symmetry"] = None
        params["Projections"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        params["Extents"] = [[-10, 10], [-10, 10], [-10, 10]]
        params["Bins"] = [201, 201, 201]

        return params
