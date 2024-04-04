import json

corelli = {
   'IPTS': '31429',
   'Runs': '324246',
   'UB' : None,
   'DetectorCalibration': '/SNS/CORELLI/shared/calibration/2022A/calibration.xml',
   'TubeCalibration': '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5',
   'MaskFile': '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/integration.xml',
   'VanadiumFile': '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/solid_angle_2p5-8.nxs',
   'FluxFile': '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/flux_2p5-8.nxs',
   'Integration':
   {
       'Cell' : 'Orthorhombic',
       'Centering' : 'Primitive',
       'Radius' : 0.25,
   }
}

topaz = {
   'IPTS': '31189',
   'Runs': '46917',
}

mandi = {
   'IPTS': '8776',
   'Runs': '10934',
}

snap = {
   'IPTS': '24179',
   'Runs': '51255',
}

demand = {
    'IPTS': '9884',
   'Experiment': '817',
   'Runs': '2',
}

wand2 = {
   'IPTS': '7776',
   'runs': '26640:26642',
}

plans = [corelli, topaz, mandi, snap, demand, wand2]
names = [k for k, v in locals().items() if v in plans]

for name, plan in zip(names, plans):
    with open('{}_reduction_plan.json'.format(name), 'w') as f:
        json.dump(plan, f, indent=4)

