"""
Baseline Models
---------------

A collection of simple benchmark models for univariate series.
"""

from typing import List, Optional, Sequence, Union

import numpy as np

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.ensemble_model import EnsembleModel
from darts.models.forecasting.forecasting_model import (
    ForecastingModel,
    GlobalForecastingModel,
)
from darts.timeseries import TimeSeries

logger = get_logger(__name__)

# Including Projected Nonlinear State Space Model
# Hideaki Shimazaki 2022.2.21
import sys
import os
#sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox_autograd/timeseries/'))
#sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox_autograd/src/'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../gaussian-toolbox_autograd/timeseries/'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../gaussian-toolbox_autograd/src/'))

#sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox/timeseries_jax/'))
#sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox/src_jax/'))


# from jax.config import config
# config.update("jax_enable_x64", True)
# from jax import numpy as jnp

# import tools
import factors
import observation_models
import state_models
import ssm_em

CONV_CRIT = 1e-6
MAX_ITER = 100

class NaiveMean(ForecastingModel):
    def __init__(self):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        self.mean_val = None

    def __str__(self):
        return "Naive mean predictor model"

    def fit(self, series: TimeSeries):
        super().fit(series)
        self.mean_val = np.mean(series.univariate_values())
        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        forecast = np.array([self.mean_val for _ in range(n)])
        return self._build_forecast_series(forecast)



class LSS(ForecastingModel):
    def __init__(self):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        #self.mean_val = None

    def __str__(self):
        return "Naive mean predictor model"

    def fit(self, series: TimeSeries):
        super().fit(series)
        #self.mean_val = np.mean(series.univariate_values())

        Dx = 1
        Dz = 4
        sm = state_models.LinearStateModel(Dz)
        om = observation_models.LinearObservationModel(Dx, Dz)  
        
        X=np.array([series.univariate_values()]).T
        ssm_em_lin = ssm_em.StateSpaceEM(X, observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        ssm_em_lin.run()
        self.ssm_em_lin = ssm_em_lin

        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        X = self.ssm_em_lin.X
        tmp = np.zeros((n,1)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        filter_test, mu_lin, std_lin = self.ssm_em_lin.predict(X_test, smoothed=False)
        forecast = np.array(mu_lin[-n:])
        print(mu_lin.shape)

        return self._build_forecast_series(forecast)


class NLSS(ForecastingModel):
    def __init__(self):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        #self.mean_val = None

    def __str__(self):
        return "Naive mean predictor model"

    def fit(self, series: TimeSeries):
        super().fit(series)
        #self.mean_val = np.mean(series.univariate_values())

        Dx = 1
        Dz = 5
        Dk = 50
        sm = state_models.LSEMStateModel(Dz, Dk, noise_z=1.0)
        om = observation_models.LinearObservationModel(Dx, Dz)  
        
        X = np.array([series.univariate_values()]).T
        self.mu_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0)
        X = (X - self.mu_X) / self.std_X   

        ssm_em_lin = ssm_em.StateSpaceEM(X, observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        ssm_em_lin.run()
        self.ssm_em_lin = ssm_em_lin

        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        X = self.ssm_em_lin.X
        tmp = np.zeros((n,1)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        filter_test, mu_lin, std_lin = self.ssm_em_lin.predict(X_test, smoothed=False)
        forecast = np.array(mu_lin[-n:])
        forecast = self.std_X * forecast + self.mu_X
        print(mu_lin.shape)

        return self._build_forecast_series(forecast)