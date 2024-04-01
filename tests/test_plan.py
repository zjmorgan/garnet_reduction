import os

from garnet.reduction.plan import ReductionPlan

filepath = os.path.dirname(os.path.abspath(__file__))

def test_runs():

    garnet_plan = ReductionPlan()

    assert garnet_plan.plan is None
    
    run_str = '345:347,349,350:352'
    
    runs = garnet_plan.runs_string_to_list(run_str)

    assert runs == [345, 346, 347, 349, 350, 351, 352]

    assert garnet_plan.runs_list_to_string(runs) == '345:347,349:352'

def test_load_plan():

    garnet_plan = ReductionPlan()
    
    reduction_plan = os.path.join(filepath, 'data/corelli_reduction_plan.json')
    
    garnet_plan.load_plan(reduction_plan)
    
    plan = garnet_plan.plan 

    assert plan is not None

    assert os.path.splitext(plan['DetectorCalibration'])[1] == '.xml'