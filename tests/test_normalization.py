import os
import shutil
import subprocess
import tempfile

from garnet.config.instruments import beamlines
from garnet.reduction.normalization import Normalization
from garnet.reduction.plan import ReductionPlan

benchmark = "shared/benchmark/norm"


def test_get_file():
    file = "/tmp/test.nxs"

    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")

    data = Normalization(rp.plan).get_file(file, ws="")

    base = "/tmp/test"
    app = "_(h,k,0)_[0,0,l]_[-10,10]_[-10,10]_[-10,10]_201x201x201"
    ext = ".nxs"
    symm = "_2_m"

    assert data == base + app + ext

    data = Normalization(rp.plan).get_file(file, ws="data")

    assert data == base + app + "_data" + ext

    rp.plan["Normalization"]["Symmetry"] = "2/m"

    data = Normalization(rp.plan).get_file(file, ws="")

    assert data == base + app + symm + ext

    data = Normalization(rp.plan).get_file(file, ws="data")

    assert data == base + app + symm + "_data" + ext


def test_corelli(tmpdir):
    config_file = "corelli_reduction_plan.yaml"
    reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
    script = os.path.abspath("./src/garnet/workflow.py")
    print("tmpdir",tmpdir)
    # reduction plan filepaths point to analysis
    print("reduction_plan",reduction_plan)
    rp = ReductionPlan()
    rp.load_plan(reduction_plan)
    saved_plan = os.path.join(tmpdir, config_file)
    print("saved_plan",saved_plan)
    rp.set_output(saved_plan)
    rp.save_plan(saved_plan)

    instrument_config = beamlines[rp.plan["Instrument"]]
    facility = instrument_config["Facility"]
    name = instrument_config["Name"]
    baseline_path = os.path.join("/", facility, name, benchmark)
    print("baseline_path",baseline_path)
    command = ["python", script, saved_plan, "norm", "3"]
    subprocess.run(command, check=False)

    #if os.path.exists(baseline_path):
    #    shutil.rmtree(baseline_path)

    #shutil.copytree(tmpdir, baseline_path)

#cannot run
# def test_topaz(tmpdir):
#     config_file = "topaz_reduction_plan.yaml"
#     reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
#     script = os.path.abspath("./src/garnet/workflow.py")

#     rp = ReductionPlan()
#     rp.load_plan(reduction_plan)
#     saved_plan = os.path.join(tmpdir, config_file)
#     print("saved_plan",saved_plan)
#     rp.set_output(saved_plan)
#     rp.save_plan(saved_plan)

#     instrument_config = beamlines[rp.plan["Instrument"]]
#     facility = instrument_config["Facility"]
#     name = instrument_config["Name"]
#     baseline_path = os.path.join("/", facility, name, benchmark)
#     print("baseline_path",baseline_path)

#     command = ["python", script, saved_plan, "norm", "2"]
#     subprocess.run(command, check=False)

        # if os.path.exists(baseline_path):
        #     shutil.rmtree(baseline_path)

        # shutil.copytree(tmpdir, baseline_path)

#laptop freezes!
# def test_demand(tmpdir):
#     config_file = "demand_reduction_plan.yaml"
#     reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
#     script = os.path.abspath("./src/garnet/workflow.py")
#     command = ["python", script, config_file, "norm", "4"]



#     rp = ReductionPlan()
#     rp.load_plan(reduction_plan)
#     saved_plan = os.path.join(tmpdir, config_file)
#     print("saved_plan",saved_plan)
#     rp.set_output(saved_plan)
#     rp.save_plan(saved_plan)

#     instrument_config = beamlines[rp.plan["Instrument"]]
#     facility = instrument_config["Facility"]
#     name = instrument_config["Name"]
#     baseline_path = os.path.join("/", facility, name, benchmark)
#     print("baseline_path",baseline_path)
#     command = ["python", script, saved_plan, "norm", "2"]
#     subprocess.run(command, check=False)

        # if os.path.exists(baseline_path):
        #     shutil.rmtree(baseline_path)

        # shutil.copytree(tmpdir, baseline_path)

#check this too
# def test_wand2(tmpdir):
#     config_file = "wand2_reduction_plan.yaml"
#     reduction_plan = os.path.abspath(os.path.join("./tests/data", config_file))
#     script = os.path.abspath("./src/garnet/workflow.py")


#     rp = ReductionPlan()
#     rp.load_plan(reduction_plan)
#     saved_plan = os.path.join(tmpdir, config_file)
#     print("saved_plan",saved_plan)
#     rp.set_output(saved_plan)
#     rp.save_plan(saved_plan)

#     instrument_config = beamlines[rp.plan["Instrument"]]
#     facility = instrument_config["Facility"]
#     name = instrument_config["Name"]
#     baseline_path = os.path.join("/", facility, name, benchmark)

#     command = ["python", script, saved_plan, "norm", "48"]
#     subprocess.run(command, check=False)

        # if os.path.exists(baseline_path):
        #     shutil.rmtree(baseline_path)

        # shutil.copytree(tmpdir, baseline_path)
