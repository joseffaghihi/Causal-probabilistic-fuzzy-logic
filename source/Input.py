#!/usr/bin/env python
__author__ = "Youssef Barkaoui"
__email__ = "barkaouionline@gmail.com"



import numpy as np
import skfuzzy as fuzz
import pandas as pd
from pandas._typing import FrameOrSeries
from scipy import stats
import matplotlib.pyplot as plt

class Fuzzy_input():
    """
        class aiming to manipulate vectors and tensors and
        return the fuzzified  version of it's input
    """

    def __init__(self):
        pass

    @staticmethod
    def attributes(data):
        max = int(data.max())
        min = int(data.min())
        med = int((min + max) / 2)
        out = [min, med, max]
        return out

    def fuzzify(self, data: FrameOrSeries, distribution: str,plot= False) -> FrameOrSeries:
        """
            Class method aiming to fuzzify only on data frame serie at once
            The output of every single serie will be appended to one fuzzified
            dataframe containing all equivalent fuzzified dataframe series.

            Parameters:
            ----------

            data : Pandas DataFrame, serie to fuzzify.
            distribution :  string ,to precise the fuzzy function distribution triangular or gaussian.
            plot : boolean to plot , False by default .

            Returns:
            -------

            out : list of fuzzy sets.

            Notes
            -----
            The process of precising the Closure of the given data and the number of fuzzy sets is automated ,
            Thus the User is left to input only the distribution of the fuzzy membership function.


            Examples
            --------
            >>> df= pd.DataFrame({'Data':np.arange(5)})
            # Generate fuzzy membership
            >>> Test=Fuzzy_only()
            >>> out = Test.fuzzify(df,'gaussian')
            >>> out
            [array([[1.        ],
               [0.81873075],
               [0.44932896],
               [0.16529889],
               [0.0407622 ]]), array([[0.44932896],
               [0.81873075],
               [1.        ],
               [0.81873075],
               [0.44932896]]), array([[0.0407622 ],
               [0.16529889],
               [0.44932896],
               [0.81873075],
               [1.        ]])]



             """
        data = np.asanyarray(data)
        param = Fuzzy_input.attributes(data)
        if distribution == 'triangular':
            set_1 = fuzz.trimf(data, [param[0], param[0], param[1]])
            set_2 = fuzz.trimf(data, [param[0], param[1], param[2]])
            set_3 = fuzz.trimf(data, [param[1], param[2], param[2]])
        elif distribution == 'gaussian':
            set_1 = fuzz.gaussmf(data, param[0], stats.tstd(data))
            set_2 = fuzz.gaussmf(data, param[1], stats.tstd(data))
            set_3 = fuzz.gaussmf(data, param[2], stats.tstd(data))

        else :
            raise Exception("Only the triangular and gaussian are accepted")
        if plot==True:
            fig, ax0 = plt.subplots(nrows=1, figsize=(3, 3))

            ax0.plot(data, set_1, 'b', linewidth=1.5, label='Poor')
            ax0.plot(data, set_2, 'g', linewidth=1.5, label='Average')
            ax0.plot(data, set_3, 'r', linewidth=1.5, label='Great')
            ax0.set_title('Data Fuzzy  set')
            ax0.legend()
            plt.show()
        return [set_1, set_2, set_3]

    def fuzzify_all(self, data: FrameOrSeries, distribution: str) -> FrameOrSeries:
        """
          A function that fuzzifies mutliple variables at once

        :Parameters
        ----------

        data  :  pandas DataFrame , The tensor to fuzzify containing the
                                            variables columns names and values.

        distribution :  string , to precise the fuzzy function distribution triangular or gaussian.

        :Returns
        -------
        A fuzzified DataFrame of the input data

        See Also
        -------
        fuzzify could be useful in the case of 1D vector

        """
        if isinstance(data, pd.DataFrame):
            fuzzy_data = pd.DataFrame()
            for column in data:
                col = pd.DataFrame({column + '_low': self.fuzzify(data[column],distribution)[0],
                                    column + '_average': self.fuzzify(data[column],distribution)[1],
                                    column + '_high': self.fuzzify(data[column],distribution)[2]})
                fuzzy_data = pd.concat([fuzzy_data, col], axis=1)
            return fuzzy_data
