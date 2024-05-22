import sys
import os
import concurrent.futures

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, ".."))
sys.path.append(directory)

from mantid import config

config["Q.convention"] = "Crystallography"

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.parallel import ParallelTasks
from garnet.reduction.integration import Integration
from garnet.reduction.normalization import Normalization

inst_dict = {
    "corelli": "CORELLI",
    "bl9": "CORELLI",
    "topaz": "TOPAZ",
    "bl12": "TOPAZ",
    "mandi": "MANDI",
    "bl11b": "MANDI",
    "snap": "SNAP",
    "bl3": "SNAP",
    "demand": "DEMAND",
    "hb3a": "DEMAND",
    "wand2": "WAND²",
    "hb2c": "WAND²",
}


def delete_directory(path):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for root, dirs, files in os.walk(path, topdown=False):
            executor.map(os.remove, [os.path.join(root, name) for name in files])
        os.rmdir(path)


if __name__ == "__main__":
    filename, reduction, arg = sys.argv[1], sys.argv[2], sys.argv[3]

    if arg.isdigit():
        n_proc = int(arg)
    else:
        instrument = inst_dict[arg.lower()]
        assert filename.endswith(".yaml")

    rp = ReductionPlan()

    if reduction == "temp":
        rp.generate_plan(instrument)
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            rp.plan.pop("OutputName", None)
            rp.plan.pop("OutputPath", None)
            rp.save_plan(filename)

    else:
        rp.load_plan(filename)

        if reduction == "int":
            func = Integration.integrate_parallel
            comb = Integration.combine_parallel
            path = "integration"
        elif reduction == "norm":
            func = Normalization.normalize_parallel
            comb = Normalization.combine_parallel
            path = "normalization"

        output = os.path.join(rp.plan["OutputPath"], path)
        if not os.path.exists(output):
            os.mkdir(output)

        output = os.path.join(rp.plan["OutputPath"], path, "plots")
        if os.path.exists(output):
            delete_directory(output)
        os.mkdir(output)

        output = os.path.join(rp.plan["OutputPath"], path, "diagnostics")
        if os.path.exists(output):
            delete_directory(output)
        os.mkdir(output)

        pt = ParallelTasks(func, comb)

        max_proc = min(os.cpu_count(), len(rp.plan["Runs"]))

        if n_proc > max_proc:
            n_proc = max_proc

        pt.run_tasks(rp.plan, n_proc)
