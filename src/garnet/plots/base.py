import matplotlib.pyplot as plt

class BasePlot:

    def __init__(self):
        """
        Create a plot.

        """

        plt.close('all')

        self.fig = plt.figure()

    def save_plot(self, filename):
        """
        Save plot.

        Parameters
        ----------
        filename : str
            Path to file.

        """

        self.fig.savefig(filename, bbox_inches='tight')