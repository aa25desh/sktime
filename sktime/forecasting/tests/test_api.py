#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

import numpy as np
import pytest
from pytest import raises
from sktime.datasets import load_airline
from sktime.forecasting.api import Forecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.utils.validation.forecasting import check_fh

DEFAULT_WINDOW_LENGTHS = [1, 5]
DEFAULT_STEP_LENGTHS = [1, 5]
DEFAULT_OOS_FHS = [1, np.array([2, 5])]
DEFAULT_INS_FHS = [
    -3,  # single in-sample
    np.array([-2, -5]),  # multiple in-sample
    0,
    # np.array([-3, 2])  # mixed in-sample and out-of-sample
]
DEFAULT_MIXED_FHS = [
    [-5, 2],
    [5, -2],
    [0, 1]
]
DEFAULT_SPS = [3, 7, 12]


y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=.3)


@pytest.mark.parametrize("fh", DEFAULT_OOS_FHS)
def test_predict_oos(fh):
    f = Forecaster()
    f.fit(y_train)
    y_pred = f.predict(fh)
    np.testing.assert_array_equal(y_pred, np.repeat(y_train.iloc[-1], len(check_fh(fh))))


@pytest.mark.parametrize("fh", DEFAULT_INS_FHS)
def test_predict_ins(fh):
    f = Forecaster()
    f.fit(y_train)
    y_pred = f.predict(fh)
    np.testing.assert_array_equal(y_pred, y_train.iloc[check_fh(fh) + y_train.index[-1] - 1].values)


@pytest.mark.parametrize("fh", DEFAULT_MIXED_FHS)
def test_predict_mixed(fh):
    f = Forecaster()
    f.fit(y_train)
    y_pred = f.predict(fh)


def test_predict_ins_negative_abs_fh():
    f = Forecaster()
    f.fit(y_train)
    fh = np.array([-len(y_train) - 1])
    with raises(ValueError):
        f.predict(fh)


@pytest.mark.parametrize("fh", DEFAULT_OOS_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_update_predict_oos_y_test(fh, window_length, step_length):
    f = Forecaster()
    f.fit(y_train)
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)
    y_pred = f.update_predict(y_test, cv)


@pytest.mark.parametrize("fh", DEFAULT_INS_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_update_predict_ins_y_test(fh, window_length, step_length):
    f = Forecaster()
    f.fit(y_train)
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)
    y_pred = f.update_predict(y_test, cv)


@pytest.mark.parametrize("fh", DEFAULT_OOS_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_update_predict_oos_y_train(fh, window_length, step_length):
    f = Forecaster()
    f.fit(y_train)
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)
    y_pred = f.update_predict(y_train, cv)


@pytest.mark.parametrize("fh", DEFAULT_INS_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_update_predict_ins_y_train(fh, window_length, step_length):
    f = Forecaster()
    f.fit(y_train)
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)
    y_pred = f.update_predict(y_train, cv)


@pytest.mark.parametrize("fh", DEFAULT_MIXED_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_update_predict_mixed_y_test(fh, window_length, step_length):
    f = Forecaster()
    f.fit(y_train)
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)
    y_pred = f.update_predict(y_test, cv)


@pytest.mark.parametrize("fh", DEFAULT_MIXED_FHS)
@pytest.mark.parametrize("window_length", DEFAULT_WINDOW_LENGTHS)
@pytest.mark.parametrize("step_length", DEFAULT_STEP_LENGTHS)
def test_update_predict_mixed_y_train(fh, window_length, step_length):
    f = Forecaster()
    f.fit(y_train)
    cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length=step_length)
    y_pred = f.update_predict(y_train, cv)


