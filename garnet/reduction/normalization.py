class Normalization:

    def __init__(self, reduction_plan, norm_plan):

        self.reduction_plan = reduction_plan


class NormalizationPlan:

    def __init__(self):

        self.name = 'Na2Al2Si3O10_H2O_2_300K'
        
        self.projections = [[1,0,0],[0,1,0],[0,0,1]]
        self.extents     = [-10,10],[-36,36],[-12,12]
        self.bins        = [201,721,241]
            
def load_config(filename):

    return NormalizationPlan()