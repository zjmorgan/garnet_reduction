import sys
import os

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, '..'))
sys.path.append(directory)

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.parallel import ParallelTasks
from garnet.reduction.integration import Integration

filename, n_proc = sys.argv[1], int(sys.argv[2])

if __name__ == '__main__':

    rp = ReductionPlan()

    rp.load_plan(filename)
    
    pt = ParallelTasks(Integration.integrate_parallel,
                       Integration.combine_parallel)

    pt.run_tasks(rp.plan, n_proc)