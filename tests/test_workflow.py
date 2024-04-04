from garnet.reduction.plan import ReductionPlan
from garnet.reduction.parallel import ParallelTasks
from garnet.reduction.integration import Integration

file = '/SNS/CORELLI/IPTS-29639/shared/integration/YbMnSb2_Pnma_100k_0p0GPa_released.json'
n_proc = 12

if __name__ == '__main__':

    rp = ReductionPlan()

    rp.load_plan(file)
    
    pt = ParallelTasks(Integration.integrate_parallel,
                       Integration.combine_parallel)

    pt.run_tasks(rp.plan, n_proc)