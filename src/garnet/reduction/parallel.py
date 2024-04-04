import os

import multiprocessing
import numpy as np

from mantid import config

class ParallelTasks:

    def __init__(self, function, combine=None):

        self.function = function
        self.combine = combine
        self.results = None

    def run_tasks(self, plan, n_proc):
        """
        Run parallel tasks with processing pool.

        Parameters
        ----------
        plan : dict
            Data reduction plan split over each process.
        n_proc : int
            Number of processes.

        """

        runs = plan['Runs']

        split = [split.tolist() for split in np.array_split(runs, n_proc)]

        join_args = [(plan, s, proc) for proc, s in enumerate(split)]

        config['MultiThreaded.MaxCores'] == 1
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['_SC_NPROCESSORS_ONLN'] = '1'

        multiprocessing.set_start_method('spawn', force=True)    
        with multiprocessing.get_context('spawn').Pool(n_proc) as pool:
            self.result = pool.starmap(self.function, join_args)
            pool.close()
            pool.join()

        if self.combine is not None:

            self.combine(plan, self.results)