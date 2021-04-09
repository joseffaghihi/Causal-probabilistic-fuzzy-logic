import numpy as np


class Score(object):
    """
        Class aiming to evaluate the causal effect metrics

    """

    def __init__(self, y, t, y_cf=None, mu0=None, mu1=None)  :
        """
            Class aiming to evaluate the causal effect metrics
            ate :  average treatment effect
            ite : individual treatment effect

            Parameters:
            -----------
            y : list  ,the outcome values list
            t : list  , the treatment binary values list
            y_cf : list , the counterfactual values list
            mu0 : list
            mu1 : list

        """
        self.y = y
        self.t = t
        self.y_cf = y_cf
        self.mu0 = mu0
        self.mu1 = mu1
        if mu0 is not None and mu1 is not None:
            self.true_ite = mu1 - mu0

    def ite(self, ypred1, ypred0):
        pred_ite = np.zeros_like(self.true_ite)
        idx1, idx0 = np.where(self.t == 1), np.where(self.t == 0)
        ite1, ite0 = self.y[idx1] - ypred0[idx1], ypred1[idx0] - self.y[idx0]
        pred_ite[idx1] = ite1
        pred_ite[idx0] = ite0
        return np.sqrt(np.mean(np.square(self.true_ite - pred_ite)))

    def ate(self, ypred1, ypred0):
        """

        Parameters:
        ----------
        ypred1
        ypred0

        Returns:
        -------
        out : float , the real  ate corresponding to mu1-mu0

        """
        return np.abs(np.mean(ypred1 - ypred0) - np.mean(self.true_ite))



    def evaluate(self, ypred1, ypred0) -> tuple :
        """
            method aiming to compute the causal effect metrics
            ate :  average treatment effect
            ite : individual treatment effect

            Parameters :
            ----------
            ypred1 : list ,outcome when treatment is given
            ypred0 : list , outcome of the control group

            Returns:
            -------

            output : tuple , (ite, ate)

        """
        ite = self.ite(ypred1, ypred0)
        ate = self.ate(ypred1, ypred0)
        return ite, ate





