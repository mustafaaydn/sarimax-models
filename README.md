# sarimax-models
Estimate S-ARIMA-X models with Stochastic Gradient Descent or Kalman Filter

### What is S-ARIMA-X?

It stands for *Seasonal AutoRegressive Integrated Moving Average with eXogenous variables*...

The formulation is:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;
![sarimax_eqn](img/model_equation.svg)

### Purpose of the repository

We need to estimate those greek letters, i.e., the S-AR, S-MA, X parameters of the model, which is said to be of order *(p, d, q)(P, D, Q)[m]*.

First way is to use Stochastic Gradient Descent (SGD). Logic is this: we fit a large AR(p) model first, get the residuals of it and use it as the unobserved noise in the formulation. Then we form the design matrix and employ SGD to find all the parameters.

Second way is via MLE: we put the S-ARIMA-X into a (Hamiltonian) state space formulation, run a Kalman filter for likelihood and a numerical optimizer solves for parameters (e.g, L-BFGS).

### Example result
Here is a sample monthly data from M4 Competiton dataset (#N1944). Black vertical line is train-test split point. All models have the order (1, 1, 1)(1, 0, 1)[12]:

![m4_data_comparison](img/example_run.png)

The Kalman filter way closely follows `statsmodels.tsa.statespace.SARIMAX` and this is because statsmodels also estimates via MLE with Kalman Filter.

### Usage

I tried to write a scikit-learn'esque interface.

For SGD:

```python
from sgd_sarimax import SARIMAX_SGD
model = SARIMAX_SGD(endog, exog, order=(2, 1, 3), seas_order=(1, 1, 0, 4))
model_fit = model.fit(max_iter=5_000, eta0=0.1)

preds_in_sample = model.predict_in_sample()
foreacasts = model.forecast(steps=10)
```

For KF:

```python
from kalman_sarimax import SARIMAX_KF
model = SARIMAX_KF(endog, exog, order=(2, 1, 3), seas_order=(1, 1, 0, 4))
model_fit = model.fit()

preds_in_sample = model.predict_in_sample()
foreacasts = model.forecast(steps=10)
```
