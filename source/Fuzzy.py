import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from pandas._typing import FrameOrSeries
from source.Input import Fuzzy_input


class Fuzzy():
    '''
    intialize the fuzzy class '''


    def __init__(self):
        pass

    def fuzzify(self, data : FrameOrSeries,cont_feats:list,potencial_index:int, Target_index:int) -> FrameOrSeries :
        """
            Class method aiming to fuzzify only on data frame serie at once
            The output of every single serie will be appended to one fuzzified
            dataframe containing all equivalent fuzzified dataframe series.

            Parameters:
            ----------

            data : Pandas DataFrame, serie to fuzzify.
            cont_feats: list , precising which variables are continuous to an index list.
            potencial_index : int , index of the variable to to estimate it's effect(eq. to treatment)
            Target_index : int, index of the variable to search it's causes(eq. to outcome)

            Returns:
            -------

            out : list of fuzzy sets.

            Notes
            -----
            The process of precising the Closure of the given data and the number of fuzzy sets is automated ,
            This class adds also the automoation of numbers of fuzzy sets using  fuzzy C-means Clustering testing
            multiple outcomes with a number of clusters ranging from 2 to 9 and using the optimal number of fuzzy sets
            corresponding to the highest fuzzy partition coefficient(In case if the input data has more than two
            continuous variables so it is possible to apply the clustering).

        """
        ##########################
        #Fuzzy c-means clustering#
        ##########################
        if len(cont_feats)>1 and isinstance(data, pd.DataFrame) :
            np.random.seed(10)
            xpts=np.asanyarray(data.iloc[:,0])
            ypts=np.asanyarray(data.iloc[:,1])

            colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

            fig1, axes1 = plt.subplots(3, 3, figsize=(8, 8))
            alldata = np.vstack((xpts, ypts))
            fpcs = []

            for ncenters, ax in enumerate(axes1.reshape(-1), 2):
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    alldata, ncenters, 2, error=0.005, maxiter=1000, init=None)

                # Store fpc values for later
                fpcs.append(fpc)

                # Plot assigned clusters, for each data point in training set
                cluster_membership = np.argmax(u, axis=0)
                for j in range(ncenters):
                    ax.plot(xpts[cluster_membership == j],
                            ypts[cluster_membership == j], '.', color=colors[j])





            fig2, ax2 = plt.subplots()
            ax2.plot(np.r_[2:11], fpcs)
            ax2.set_xlabel("Number of centers")
            ax2.set_ylabel("Fuzzy partition coefficient")


            #############################################################
            #Building the model With Argmax(Fuzzy partition Coefficient)#
            #-----------------------------------------------------------#
            #############################################################

            cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
                alldata, 3, 2, error=0.005, maxiter=1000)

            # Show 3-cluster model
            fig2, ax2 = plt.subplots()
            ax2.set_title('Trained model')
            for j in range(3):
                ax2.plot(alldata[0, u_orig.argmax(axis=0) == j],
                         alldata[1, u_orig.argmax(axis=0) == j], 'o',
                         label='series ' + str(j))
            ax2.legend()



            # Generate uniformly sampled data spread across the range [0, 10] in x and y
            newdata = np.random.uniform(0, 1, (1100, 2)) * 10

            u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                newdata.T, cntr, 2, error=0.005, maxiter=1000)


            cluster_membership = np.argmax(u, axis=0)


        else :
            raise Exception("Not Enough Continuous Variable to Cluster")
        if np.argmax(fpcs)>1 and np.argmax(fcps)%2==1 :
            num_set=np.argmax(fpcs)
        else :
            num_set=3
        bin_feats=[i for i in range(len(data.columns)) if i not in cont_feats ]
        df=data.copy()
        out=pd.DataFrame()
        for column in df.columns:
            if df.columns.get_loc(column) in cont_feats:
                obj = Fuzzy_input()
                col = pd.DataFrame({column + '_low': obj.fuzzify(df[column], 'gaussian')[0],
                                    column + '_average': obj.fuzzify(df[column], 'gaussian')[1],
                                    column + '_high': obj.fuzzify(df[column], 'gaussian')[2]})
                out = pd.concat([out, col], axis=1)
            else :
                out = pd.concat([out,df[column]],axis=1)

        return out
















