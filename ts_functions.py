from __future__ import division
import numpy as np


def single_moving_average (series, window):
    """
    Smooths a time series via single moving average method over 
    a particular window of time. In this method, previous observations
    are assigned equal weight.
    Inputs:
        series - time series to smooth
        window - number of time periods to use for averaging
    """

    smoothed_series = []

    for i in xrange(window - 1):
        smoothed_series.append(np.NaN)
    for i in xrange(len(series) - window + 1):
        window_values = series[i:i + window]
        smoothed = sum(window_values) / float(window)
        smoothed_series.append(smoothed)

    return smoothed_series


def exponential_smoothing (series, alpha):
    """
    Smooths a time series via exponential moving average method. In this
    method, previous observations are assigned exponentially decreasing
    weights.
    Inputs:
        series - time series to smooth
        alpha - smoothing constant; 0 < alpha <= 1
    """

    smoothed_series = [np.NaN]          # there is no S1
    smoothed_series.append(series[0])   # initialize with S2 = y1

    for i in xrange(2, len(series)):
        smoothed = alpha * series[i - 1] + (1 - alpha) * smoothed_series[-1]
        smoothed_series.append(smoothed)

    return smoothed_series


def double_exponential_smoothing (series, alpha, gamma):
    """
    Smooths a time series via double exponential smoothing method. In this
    method, previous observations are assigned exponentially decreasing
    weights. This method is better at following trends in data than single
    exponential smoothing.
    Inputs:
        series - time series to smooth
        alpha - smoothing constant; 0 < alpha <= 1
        gamma - second smoothing constant; 0 < gamma <= 1
    """

    smoothed_series = [series[0]]               # initialize with S1 = y1
    double_smoothed = [series[1] - series[0]]   # initialize with b1 = y2 - y1
    # can also be initialized by other methods

    for i in xrange(1, len(series)):
        smoothed = alpha * series[i] + (1 - alpha) * (smoothed_series[-1] + double_smoothed[-1])
        double_smooth = gamma * (smoothed - smoothed_series[-1]) + (1 - gamma) * double_smoothed[-1]

        smoothed_series.append(smoothed)
        double_smoothed.append(double_smooth)

    return smoothed_series


def lasp_forecast (series, alpha, gamma, periods):
    """
    Forecasts a time series via double exponential smoothing method, aka LASP.
    Inputs:
        series - time series to smooth
        alpha - smoothing constant; 0 < alpha <= 1
        gamma - second smoothing constant; 0 < gamma <= 1
        periods - number of time periods to forecast
    """

    forecast_series = [series[-1]]
    smoothed_series = [series[0]]               # initialize with S1 = y1
    double_smoothed = [series[1] - series[0]]   # initialize with b1 = y2 - y1
    # can also be initialized by other methods

    for i in xrange(1, len(series)):
        smoothed = alpha * series[i] + (1 - alpha) * (smoothed_series[-1] + double_smoothed[-1])
        double_smooth = gamma * (smoothed - smoothed_series[-1]) + (1 - gamma) * double_smoothed[-1]
        smoothed_series.append(smoothed)
        double_smoothed.append(double_smooth)

    for i in xrange(periods):
        forecast = forecast_series[-1] + double_smoothed[-1]
        forecast_series.append(forecast)

    return forecast_series[1:]

