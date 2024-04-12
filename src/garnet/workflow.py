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

filename, reduction, arg = sys.argv[1], sys.argv[2], sys.argv[3]

inst_dict = {'corell': 'CORELLI', 
             'bl9': 'CORELLI',
             'topaz': 'TOPAZ',
             'bl12': 'TOPAZ',
             'mandi': 'MANDI',
             'bl11b': 'MANDI',
             'snap': 'SNAP',
             'bl3': 'SNAP',
             'demand': 'DEMAND',
             'hb3a': 'DEMAND',
             'wand2': 'WAND²',
             'hb2c': 'WAND²'}

if type(arg) is int:
    n_proc = int(arg)
else:
    instrument = inst_dict[arg.lower()]
    assert filename.endswith('.yaml')

if __name__ == '__main__':

    rp = ReductionPlan()

    if reduction == 'temp':

        rp.generate_plan(instrument)
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            rp.save_plan(filename)

    else:

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