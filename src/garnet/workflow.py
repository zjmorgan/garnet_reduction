import sys
import os

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..'))
sys.path.append(directory)

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.parallel import ParallelTasks
from garnet.reduction.integration import Integration
from garnet.reduction.normalization import Normalization

filename, reduction, n_proc = sys.argv[1], sys.argv[2], int(sys.argv[3])

if __name__ == '__main__':

    rp = ReductionPlan()

    rp.load_plan(filename)

    if reduction == 'int':
        func = Integration.integrate_parallel
        comb = Integration.combine_parallel
    elif reduction == 'norm':
        func = Normalization.normalize_parallel
        comb = Normalization.combine_parallel

    pt = ParallelTasks(func, comb)

    max_proc = os.cpu_count()

    if n_proc > max_proc:
        n_proc = max_proc

    pt.run_tasks(rp.plan, n_proc)