import json

corelli = {
  'IPTS': '31429',
  'Runs': '324246',
  'DetectorCalibration': '/SNS/CORELLI/shared/calibration/2022A/calibration.xml',
  'TubeCalibration': '/SNS/CORELLI/shared/calibration/tube/calibration_corelli_20200109.nxs.h5',
  'MaskFile': '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/integration.xml',
  'VanadiumFile': '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/solid_angle_2p5-8.nxs',
  'FluxFile': '/SNS/CORELLI/shared/Vanadium/2022B_0725_CCR_5x7/flux_2p5-8.nxs',
}

topaz = {
  'IPTS': '31189',
  'Runs': '46917',
}


plans = [corelli, topaz]
names = [k for k, v in locals().items() if v in plans]

for name, plan in zip(names, plans):
    with open('{}_reduction_plan.json'.format(name), 'w') as f:
        json.dump(plan, f, indent=4)

