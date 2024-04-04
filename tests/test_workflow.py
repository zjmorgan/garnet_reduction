from garnet.reduction.plan import ReductionPlan
from garnet.reduction.integration import Integration

reduction_plan = '/SNS/CORELLI/IPTS-29639/shared/integration/YbMnSb2_Pnma_100k_0p0GPa_released.json'

garnet_plan = ReductionPlan()
garnet_plan.load_plan(reduction_plan)

files = ['/SNS/CORELLI/IPTS-29639/shared/integration/integration/YbMnSb2_Pnma_100k_0p0GPa_released_p{}.nxs'.format(i) for i in range(16)]

Integration(garnet_plan.plan).combine(files)