import sys
import os

sys.path.append(os.path.expanduser('~/.git/garnet_reduction/'))

from garnet.reduction import configuration, normalization

def test_run_normalization_config():

    config_path = os.path.join('data', 'CORELLI_normalization.config')

    # generated from tab 1
    reduction_plan = configuration.load_config(config_path)
    # generated from tab 3
    norm_plan = normalization.load_config(config_path)

    # run the reduction from tab3
    norm = normalization.Normalization(reduction_plan, norm_plan)

    norm.run()
    
    UB = norm.UB
    
    reduction_plan.update_UB(UB)
    reduction_plan.update_with_normalization_plan(norm_plan)
    
    assert reduction_plan.UB == UB