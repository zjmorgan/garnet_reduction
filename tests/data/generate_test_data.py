from garnet.reduction.plan import ReductionPlan, save_YAML

corelli = ReductionPlan()
corelli.generate_plan('CORELLI')
corelli.plan['IPTS'] = 31429
corelli.plan['Runs'] = '324246'
corelli.plan['UBFile'] = None
corelli.plan['DetectorCalibration'] = '/SNS/CORELLI/shared/calibration/2022A/calibration.xml'
corelli.plan['MaskFile'] = '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/integration.xml'
corelli.plan['VanadiumFile'] = '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/solid_angle_2p5-8.nxs'
corelli.plan['FluxFile'] = '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/flux_2p5-8.nxs'
corelli.plan['Integration']['Cell'] = 'Cubic'
corelli.plan['Integration']['Centering'] = 'I'
corelli.plan['Integration']['MinD'] = 0.7
corelli.plan['Integration']['Radius'] = 0.25
corelli.plan['Normalization']['Symmetry'] = None

topaz = ReductionPlan()
topaz.generate_plan('TOPAZ')

mandi = ReductionPlan()
mandi.generate_plan('MANDI')

snap = ReductionPlan()
snap.generate_plan('SNAP')

demand = ReductionPlan()
demand.generate_plan('DEMAND')

wand2 = ReductionPlan()
wand2.generate_plan('WAND²')

# topaz = {
#     'Instrument': 'TOPAZ',
#     'IPTS': '31189',
#     'Runs': '46917',
# }

# mandi = {
#     'Instrument': 'MANDI',
#     'IPTS': '8776',
#     'Runs': '10934',
# }

# snap = {
#     'Instrument': 'SNP',
#     'IPTS': '24179',
#     'Runs': '51255',
# }

# demand = {
#     'Instrument': 'DEMAND',
#     'IPTS': '9884',
#     'Experiment': '817',
#     'Runs': '2',
# }

# wand2 = {
#     'Instrument': 'WAND²',
#     'IPTS': '7776',
#     'runs': '26640:26642',
# }

plans = [corelli, topaz, mandi, snap, demand, wand2]
names = [k for k, v in locals().items() if v in plans]

for name, plan in zip(names, plans):
    save_YAML(plan.plan, '{}_reduction_plan.yaml'.format(name))
