import os
import tempfile
import shutil
import subprocess

from garnet.reduction.plan import ReductionPlan
from garnet.config.instruments import beamlines

benchmark = 'shared/benchmark/norm'

def test_corelli():

    config_file = 'corelli_reduction_plan.yaml'
    reduction_plan = os.path.abspath(os.path.join('./tests/data', config_file))
    script = os.path.abspath('./src/garnet/workflow.py')
    command = ['python', script, config_file, 'norm', '16']

    with tempfile.TemporaryDirectory() as tmpdir:

        os.chdir(tmpdir)

        rp = ReductionPlan()
        rp.load_plan(reduction_plan)
        rp.save_plan(os.path.join(tmpdir, config_file))

        subprocess.run(command)

        instrument_config = beamlines[rp.plan['Instrument']]
        facility = instrument_config['Facility']
        name = instrument_config['Name']
        baseline_path = os.path.join('/', facility, name, benchmark)

        subprocess.run(command)

        if os.path.exists(baseline_path):
            shutil.rmtree(baseline_path)

        shutil.copytree(tmpdir, baseline_path)

def test_demand():

    config_file = 'demand_reduction_plan.yaml'
    reduction_plan = os.path.abspath(os.path.join('./tests/data', config_file))
    script = os.path.abspath('./src/garnet/workflow.py')
    command = ['python', script, config_file, 'norm', '4']

    with tempfile.TemporaryDirectory() as tmpdir:

        os.chdir(tmpdir)

        rp = ReductionPlan()
        rp.load_plan(reduction_plan)
        rp.save_plan(os.path.join(tmpdir, config_file))

        subprocess.run(command)

        instrument_config = beamlines[rp.plan['Instrument']]
        facility = instrument_config['Facility']
        name = instrument_config['Name']
        baseline_path = os.path.join('/', facility, name, benchmark)

        subprocess.run(command)

        if os.path.exists(baseline_path):
            shutil.rmtree(baseline_path)

        shutil.copytree(tmpdir, baseline_path)

def test_wand2():

    config_file = 'wand2_reduction_plan.yaml'
    reduction_plan = os.path.abspath(os.path.join('./tests/data', config_file))
    script = os.path.abspath('./src/garnet/workflow.py')
    command = ['python', script, config_file, 'norm', '48']

    with tempfile.TemporaryDirectory() as tmpdir:

        os.chdir(tmpdir)

        rp = ReductionPlan()
        rp.load_plan(reduction_plan)
        rp.save_plan(os.path.join(tmpdir, config_file))

        instrument_config = beamlines[rp.plan['Instrument']]
        facility = instrument_config['Facility']
        name = instrument_config['Name']
        baseline_path = os.path.join('/', facility, name, benchmark)

        subprocess.run(command)

        if os.path.exists(baseline_path):
            shutil.rmtree(baseline_path)

        shutil.copytree(tmpdir, baseline_path)
