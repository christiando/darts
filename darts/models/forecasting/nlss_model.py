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

from copy import deepcopy

logger = get_logger(__name__)

# Including Projected Nonlinear State Space Model
# Hideaki Shimazaki 2022.2.21
import sys
import os
# sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox_autograd/timeseries/'))
# sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox_autograd/src/'))
#sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox/timeseries_jax/'))
#sys.path.append(os.path.join(os.getcwd(),'../gaussian-toolbox/src_jax/'))

# sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../gaussian-toolbox_old/timeseries_jax/'))
# sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../gaussian-toolbox_old/src_jax/'))

# sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../gaussian-toolbox/timeseries_jax/'))
# sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../gaussian-toolbox/src_jax/'))
# sys.path.append(os.path.join(os.path.dirname(__file__),'../../../../gaussian-toolbox/'))

# from gaussian_toolbox import timeseries
# from gaussian_toolbox.timeseries import state_model, observation_model, ssms
import timeseries_models
from timeseries_models import state_model, observation_model, state_space_model

from jax import config
config.update("jax_enable_x64", True)

from jax import numpy as jnp

# import tools
# import factors
# import observation_model
# import state_model
# import ssm_em
# from src_jax import densities

CONV_CRIT = 1e-5 #1e-6
MAX_ITER = 100 #16

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
    def __init__(self, Dx: int = 1, Dz: int = 1):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        self.Dx = Dx
        self.Dz = Dz

    def __str__(self):
        return "LSS model"

    def fit(self, series: TimeSeries):
        super().fit(series)

        sm = state_model.LinearStateModel(self.Dz)
        om = observation_model.LinearObservationModel(self.Dx, self.Dz)  
        #om.pca_init(X)

        X = np.array([series.univariate_values()]).T
        self.mu_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0)
        X = (X - self.mu_X) / self.std_X   

        ssm_em_lin = state_space_model.StateSpaceModel(jnp.array(X), observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        ssm_em_lin.fit()
        self.ssm_em_lin = ssm_em_lin

        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        X = self.ssm_em_lin.X
        tmp = np.zeros((n,1)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        filter_test, mu_lin, std_lin = self.ssm_em_lin.predict(jnp.array(X_test), smoothed=False)['x']
        forecast = np.array(mu_lin[-n:])
        forecast = self.std_X * forecast + self.mu_X
        print(mu_lin.shape)

        return self._build_forecast_series(forecast)


class LSS_Takens(ForecastingModel):
    def __init__(self, Dx: int = 1, Dz: int = 1):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        Dx = Dz
        self.Dx = Dx
        self.Dz = Dz
        print(f"Dx = {Dx}, Dz = {Dz}")

    def __str__(self):
        return "LSS model with Takens"

    def fit(self, series: TimeSeries):
        super().fit(series)

        sm = state_model.LinearStateModel(self.Dz)
        om = observation_model.LinearObservationModel(self.Dx, self.Dz) 
        #om.pca_init(X)

        # making the inital noise small
        # sm = state_model.LinearStateModel(self.Dz, noise_z=0.01)
        # om = observation_model.LinearObservationModel(self.Dx, self.Dz, noise_x=0.01)  

        x = np.array([series.univariate_values()]).T
        self.mu_x = np.mean(x, axis=0)
        self.std_x = np.std(x, axis=0)
        # x = (x - self.mu_x) / self.std_x

        T = len(x)
        step = int(200/self.Dx)
        X = np.zeros((T-200+step,self.Dx))
        for i in range(self.Dx):
            X[:,i] = np.array(series.univariate_values()[i*step:T-200+(i+1)*step])

        # ssm_em_lin = state_space_model.StateSpaceModel(jnp.array(X), observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        # ssm_em_lin.fit()
        ssm_em_lin = state_space_model.StateSpaceModel(observation_model=om, state_model=sm)
        #ssm_em_lin.fit(jnp.array(X), max_iter=10, conv_crit=CONV_CRIT)
        #MAX_ITER = 5
        llk_list, p0_dict, smooth_dict, two_step_smooth_dict = ssm_em_lin.fit(X, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        
        ssm_em_lin.X = jnp.array(X)
        ssm_em_lin.smooth_dict = smooth_dict
        self.ssm_em_lin = ssm_em_lin

        return self

    
    #New code 250530
    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])
        
        horizon = 1
        prediction_steps = n
        Dx = self.Dx
        smooth_dict = self.ssm_em_lin.smooth_dict
        X = self.ssm_em_lin.X
        x_prediction = np.vstack([X, np.zeros((prediction_steps, Dx))])
        data_prediction_densities = self.ssm_em_lin.predict(X=x_prediction, first_prediction_idx=X.shape[0], mu0=smooth_dict['mu'][:1], Sigma0=smooth_dict['Sigma'][:1])['x']
        
        #x_prediction = np.zeros((prediction_steps, Dx))
        #data_prediction_densities = self.ssm_em_lin.predict(X=x_prediction, first_prediction_idx=0, mu0=smooth_dict['mu'][-1:], Sigma0=smooth_dict['Sigma'][-1:])
        
        forecast = np.array(data_prediction_densities.mu[:,-1])

        return self._build_forecast_series(forecast)                                               
                                                         
        
    def predict_oldver(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        ssm_emd_copy = deepcopy(self.ssm_em_lin)
        #ssm_emd_copy.om.Qx = 0.00001*np.eye(ssm_emd_copy.Dx) #0.00001
        ssm_emd_copy.om.update_observation_density()
        #ssm_emd_copy.sm.Qz = 0.00001*np.eye(ssm_emd_copy.Dz)
        ssm_emd_copy.sm.update_state_density()

        X = self.ssm_em_lin.X
        
        tmp = np.zeros((n,self.Dx)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        # Old method using NaN
        # filter_test, mu_lin, std_lin = self.ssm_em_lin.predict(jnp.array(X_test), smoothed=False)
        # forecast = np.array(mu_lin[-n:,-1])
        
        # New code 250530
        # horizon = 1
        # prediction_steps = 400
        # x_prediction = np.vstack([x_data, np.zeros((prediction_steps, Dx))])
        # data_prediction_densities = ssm.predict(X=x_prediction, horizon=horizon, first_prediction_idx=x_data.shape[0]-1)
        # data_prediction_densities.mu[:,-1] 
        # forecast = np.array(mu_x_predict[:,-1])
        
        # # New analytical method
        T, tmp = X.shape
        fully_observed_till = T
        # p_z_fo, mu_x_fo, std_x_fo = ssm_emd_copy.predict(jnp.array(X_test[:fully_observed_till]), observed_dims=jnp.arange(X_test.shape[1]))
        data_prediction_densities = ssm_emd_copy.predict(jnp.array(X_test[:fully_observed_till]))
        data_prediction_densities
        p0 = p_z_fo.slice(jnp.array([-1]))

        # (Optional) Making the initidal density for forecasting a sample point. 
        # Sigma = 0.00001*jnp.array([jnp.eye(ssm_emd_copy.Dz)]) # jnp.ndarray with shape [1,Dz,Dz]
        # tmp = jnp.array(X_test[fully_observed_till-1]) - ssm_emd_copy.om.d
        # tmp = jnp.dot(jnp.linalg.inv(ssm_emd_copy.om.C), tmp)  #mu=C^{-1}(y-d) # jnp.ndarray with shape [1,Dz]
        # mu = jnp.array([tmp])
        # p0 = densities.GaussianDensity(Sigma=Sigma, mu=mu)

        p_z_predict, mu_x_predict, std_x_predict = ssm_emd_copy.predict(jnp.array(X_test[fully_observed_till:]), p0=p0)
        forecast = np.array(mu_x_predict[:,-1])

        # print(X_test[fully_observed_till-1])
        # print(forecast[:3])
        # forecast = self.std_x * forecast + self.mu_x
        return self._build_forecast_series(forecast)



class NLSS(ForecastingModel):
    def __init__(self, Dx: int = 1, Dz: int = 1, Dk: int = 1):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        self.Dx = Dx
        self.Dz = Dz
        self.Dk = Dk
        print(f"Dx = {Dx}, Dz = {Dz}, Dk = {Dk}")

    def __str__(self):
        return "Projected Nonlinear State Space Model"

    def fit(self, series: TimeSeries):
        super().fit(series)

        sm = state_model.LSEMStateModel(self.Dz, self.Dk, noise_z=1.0)
        om = observation_model.LinearObservationModel(self.Dx, self.Dz)  
        #om.pca_init(X)

        X = np.array([series.univariate_values()]).T
        self.mu_X = np.mean(X, axis=0)
        self.std_X = np.std(X, axis=0)
        X = (X - self.mu_X) / self.std_X   

        # ssm_emd = state_space_model.StateSpaceModel(jnp.array(X), observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        # ssm_emd.fit()
        ssm_emd = state_space_model.StateSpaceModel(observation_model=om, state_model=sm)
        ssm_emd.fit(jnp.array(X))
        self.ssm_emd = ssm_emd

        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        X = self.ssm_emd.X
        tmp = np.zeros((n,1)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        filter_test, mu, std = self.ssm_emd.predict_nans(jnp.array(X_test), smoothed=False)
        forecast = np.array(mu[-n:])
        forecast = self.std_X * forecast + self.mu_X
        print(mu.shape)

        return self._build_forecast_series(forecast)



class NLSS_Takens(ForecastingModel):
    def __init__(self, Dx: int = 1, Dz: int = 1, Dk: int = 1):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        Dx = Dz
        self.Dx = Dx
        self.Dz = Dz
        self.Dk = Dk
        print(f"Dx = {Dx}, Dz = {Dz}, Dk = {Dk}")

    def __str__(self):
        return "Projected Nonlinear State Space Model"

    def fit(self, series: TimeSeries):
        super().fit(series)

        sm = state_model.LSEMStateModel(self.Dz, self.Dk, noise_z=1.0)
        om = observation_model.LinearObservationModel(self.Dx, self.Dz)  
        #om.pca_init(X)

        x = series.univariate_values()
        x = np.array([series.univariate_values()]).T
        self.mu_x = np.mean(x, axis=0)
        self.std_x = np.std(x, axis=0)
        #x = (x - self.mu_x) / self.std_x  

        T = len(x)
        #X = np.zeros((T+1-self.Dx,self.Dx))
        #for i in range(self.Dx):
        #    X[:,i] = np.array(series.univariate_values()[i:T+1-self.Dx+i])

        step = int(200/self.Dx)
        X = np.zeros((T-200+step,self.Dx))
        for i in range(self.Dx):
            X[:,i] = np.array(series.univariate_values()[i*step:T-200+(i+1)*step])        

        #MAX_ITER = 3
        ssm_emd = state_space_model.StateSpaceModel(observation_model=om, state_model=sm)
        llk_list, p0_dict, smooth_dict, two_step_smooth_dict = ssm_emd.fit(X, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        
        ssm_emd.X = jnp.array(X)
        ssm_emd.smooth_dict = smooth_dict
  
        self.ssm_emd = ssm_emd

        return self

    
    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)

        horizon = 1
        prediction_steps = n
        Dx = self.Dx
        smooth_dict = self.ssm_emd.smooth_dict
        X = self.ssm_emd.X
        x_prediction = np.vstack([X, np.zeros((prediction_steps, Dx))])
        data_prediction_densities = self.ssm_emd.predict(X=x_prediction, first_prediction_idx=X.shape[0], mu0=smooth_dict['mu'][:1], Sigma0=smooth_dict['Sigma'][:1])['x']
                
        forecast = np.array(data_prediction_densities.mu[:,-1])

        return self._build_forecast_series(forecast)
    
    
    def predict_old(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        #ssm_emd_copy = deepcopy(self.ssm_emd)
        ssm_emd_copy = self.ssm_emd
        
        #ssm_emd_copy.om.Qx = 0.0001*np.eye(ssm_emd_copy.Dx) #0.00001
        ssm_emd_copy.om.update_observation_density()
        #ssm_emd_copy.sm.Qz = 0.0001*np.eye(ssm_emd_copy.Dz)
        ssm_emd_copy.sm.update_state_density()

        X = self.ssm_emd.X
        tmp = np.zeros((n,self.Dx)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        # Old method using NaN
        # filter_test, mu, std = ssm_emd_copy.predict_nans(jnp.array(X_test), smoothed=False)
        # forecast = np.array(mu[-n:,self.Dx-1])
        #forecast = self.std_x * forecast + self.mu_x
        
        # # New analytical method
        T, tmp = X.shape
        fully_observed_till = T
        p_z_fo, mu_x_fo, std_x_fo = ssm_emd_copy.predict(jnp.array(X_test[:fully_observed_till]), observed_dims=jnp.arange(X_test.shape[1]))
        p0 = p_z_fo.slice(jnp.array([-1]))
        
        # (Optional) Making the initidal density for forecasting a sample point. 
        # Sigma = 0.00001*jnp.array([jnp.eye(ssm_emd_copy.Dz)]) # jnp.ndarray with shape [1,Dz,Dz]
        # tmp = jnp.array(X_test[fully_observed_till-1]) - ssm_emd_copy.om.d
        # tmp = jnp.dot(jnp.linalg.inv(ssm_emd_copy.om.C), tmp)  #mu=C^{-1}(y-d) # jnp.ndarray with shape [1,Dz]
        # mu = jnp.array([tmp])
        # p0 = densities.GaussianDensity(Sigma=Sigma, mu=mu)

        p_z_predict, mu_x_predict, std_x_predict = ssm_emd_copy.predict(jnp.array(X_test[fully_observed_till:]), p0=p0)
        forecast = np.array(mu_x_predict[:,-1])

        # # Sample based
        # num_samples = 100
        # sample_z_predict, sample_x_predict = ssm_emd_copy.sample_trajectory_static(jnp.array(X_test[fully_observed_till:]), p0=p0, num_samples=num_samples)
        # forecast = np.array( jnp.mean(sample_x_predict[:,:,self.Dx-1],axis=1) )
        
        #forecast = self.std_x * forecast + self.mu_x
        return self._build_forecast_series(forecast)

    def predict_noiseless(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        ssm_emd_copy = deepcopy(self.ssm_emd)
        ssm_emd_copy.om.Qx = 0.0001*np.eye(ssm_emd_copy.Dx) #0.00001
        ssm_emd_copy.om.update_observation_density()
        ssm_emd_copy.sm.Qz = 0.0001*np.eye(ssm_emd_copy.Dz)
        ssm_emd_copy.sm.update_state_density()

        X = self.ssm_emd.X
        tmp = np.zeros((n,self.Dx)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        # Old method using NaN
        # filter_test, mu, std = ssm_emd_copy.predict_nans(jnp.array(X_test), smoothed=False)
        # forecast = np.array(mu[-n:,self.Dx-1])
        #forecast = self.std_x * forecast + self.mu_x
        
        # # New method
        T, tmp = X.shape
        fully_observed_till = T
        p_z_fo, mu_x_fo, std_x_fo = ssm_emd_copy.predict(jnp.array(X_test[:fully_observed_till]), observed_dims=jnp.arange(X_test.shape[1]))
        p0 = p_z_fo.slice(jnp.array([-1]))

        # Analytical
        p_z_predict, mu_x_predict, std_x_predict = ssm_emd_copy.predict(jnp.array(X_test[fully_observed_till:]), p0=p0)
        forecast = np.array(mu_x_predict[:,-1])

        # # Sample based
        # num_samples = 100
        # sample_z_predict, sample_x_predict = ssm_emd_copy.sample_trajectory_static(jnp.array(X_test[fully_observed_till:]), p0=p0, num_samples=num_samples)
        # forecast = np.array( jnp.mean(sample_x_predict[:,:,self.Dx-1],axis=1) )
        
        #forecast = self.std_x * forecast + self.mu_x
        return self._build_forecast_series(forecast)


    def predict_sample(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        ssm_emd_copy = deepcopy(self.ssm_emd)
        ssm_emd_copy.om.Qx = 0.0001*np.eye(ssm_emd_copy.Dx) #0.00001
        ssm_emd_copy.om.update_observation_density()
        ssm_emd_copy.sm.Qz = 0.0001*np.eye(ssm_emd_copy.Dz)
        ssm_emd_copy.sm.update_state_density()

        X = self.ssm_emd.X
        tmp = np.zeros((n,self.Dx)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        # Old method using NaN
        # filter_test, mu, std = ssm_emd_copy.predict_nans(jnp.array(X_test), smoothed=False)
        # forecast = np.array(mu[-n:,self.Dx-1])
        #forecast = self.std_x * forecast + self.mu_x
        
        # # New method
        T, tmp = X.shape
        fully_observed_till = T
        p_z_fo, mu_x_fo, std_x_fo = ssm_emd_copy.predict(jnp.array(X_test[:fully_observed_till]), observed_dims=jnp.arange(X_test.shape[1]))
        p0 = p_z_fo.slice(jnp.array([-1]))
        print(p0)

        # Analytical
        # p_z_predict, mu_x_predict, std_x_predict = ssm_emd_copy.predict(jnp.array(X_test[fully_observed_till:]), p0=p0)
        # forecast = np.array(mu_x_predict[:,-1])

        # # Sample based
        num_samples = 100
        sample_z_predict, sample_x_predict = ssm_emd_copy.sample_trajectory(jnp.array(X_test[fully_observed_till:]), p0=p0, num_samples=num_samples)
        forecast = np.array( jnp.mean(sample_x_predict[:,:,self.Dx-1],axis=1) )
        

        #forecast = self.std_x * forecast + self.mu_x
        return self._build_forecast_series(forecast)




class RBF_Takens(ForecastingModel):
    def __init__(self, Dx: int = 1, Dz: int = 1, Dk: int = 1):
        """Naive Mean Model

        This model has no parameter, and always predicts the
        mean value of the training series.
        """
        super().__init__()
        Dx = Dz
        self.Dx = Dx
        self.Dz = Dz
        self.Dk = Dk
        print(f"Dx = {Dx}, Dz = {Dz}, Dk = {Dk}")

    def __str__(self):
        return "Projected Nonlinear State Space Model"

    def fit(self, series: TimeSeries):
        super().fit(series)

        sm = state_model.LRBFMStateModel(self.Dz, self.Dk, noise_z=1.0)
        om = observation_model.LinearObservationModel(self.Dx, self.Dz)  
        #om.pca_init(X)

        x = series.univariate_values()
        x = np.array([series.univariate_values()]).T
        self.mu_x = np.mean(x, axis=0)
        self.std_x = np.std(x, axis=0)
        #x = (x - self.mu_x) / self.std_x  

        T = len(x)
        #X = np.zeros((T+1-self.Dx,self.Dx))
        #for i in range(self.Dx):
        #    X[:,i] = np.array(series.univariate_values()[i:T+1-self.Dx+i])

        step = int(200/self.Dx)
        X = np.zeros((T-200+step,self.Dx))
        for i in range(self.Dx):
            X[:,i] = np.array(series.univariate_values()[i*step:T-200+(i+1)*step])        

        # ssm_emd = state_space_model.StateSpaceModel(jnp.array(X), observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        # ssm_emd.fit()    
        ssm_emd = state_space_model.StateSpaceModel(observation_model=om, state_model=sm)
        ssm_emd.fit(jnp.array(X))
        
        self.ssm_emd = ssm_emd

        return self

    def predict(self, n: int, num_samples: int = 1):
        super().predict(n, num_samples)
        #forecast = np.array([self.mean_val for _ in range(n)])

        #ssm_emd_copy = deepcopy(self.ssm_emd)
        ssm_emd_copy = self.ssm_emd
        
        #ssm_emd_copy.om.Qx = 0.0001*np.eye(ssm_emd_copy.Dx) #0.00001
        ssm_emd_copy.om.update_observation_density()
        #ssm_emd_copy.sm.Qz = 0.0001*np.eye(ssm_emd_copy.Dz)
        ssm_emd_copy.sm.update_state_density()

        X = self.ssm_emd.X
        tmp = np.zeros((n,self.Dx)) 
        tmp[:] = np.nan
        X_test = np.vstack([X,tmp])

        # Old method using NaN
        # filter_test, mu, std = ssm_emd_copy.predict_nans(jnp.array(X_test), smoothed=False)
        # forecast = np.array(mu[-n:,self.Dx-1])
        #forecast = self.std_x * forecast + self.mu_x
        
        # # New analytical method
        T, tmp = X.shape
        fully_observed_till = T
        p_z_fo, mu_x_fo, std_x_fo = ssm_emd_copy.predict(jnp.array(X_test[:fully_observed_till]), observed_dims=jnp.arange(X_test.shape[1]))
        p0 = p_z_fo.slice(jnp.array([-1]))
        
        # (Optional) Making the initidal density for forecasting a sample point. 
        # Sigma = 0.00001*jnp.array([jnp.eye(ssm_emd_copy.Dz)]) # jnp.ndarray with shape [1,Dz,Dz]
        # tmp = jnp.array(X_test[fully_observed_till-1]) - ssm_emd_copy.om.d
        # tmp = jnp.dot(jnp.linalg.inv(ssm_emd_copy.om.C), tmp)  #mu=C^{-1}(y-d) # jnp.ndarray with shape [1,Dz]
        # mu = jnp.array([tmp])
        # p0 = densities.GaussianDensity(Sigma=Sigma, mu=mu)

        p_z_predict, mu_x_predict, std_x_predict = ssm_emd_copy.predict(jnp.array(X_test[fully_observed_till:]), p0=p0)
        forecast = np.array(mu_x_predict[:,-1])

        # # Sample based
        # num_samples = 100
        # sample_z_predict, sample_x_predict = ssm_emd_copy.sample_trajectory_static(jnp.array(X_test[fully_observed_till:]), p0=p0, num_samples=num_samples)
        # forecast = np.array( jnp.mean(sample_x_predict[:,:,self.Dx-1],axis=1) )
        
        #forecast = self.std_x * forecast + self.mu_x
        return self._build_forecast_series(forecast)



