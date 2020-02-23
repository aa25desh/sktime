#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from contextlib import contextmanager

import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import ManualWindowSplitter
from sktime.utils.validation.forecasting import check_fh, check_y, check_cv


class Forecaster:

    def __init__(self):
        self.oh = pd.Series([])
        self.cutoff = None
        self.window_length = 1
        self.fh = None
        self.fh_abs = None

    def fit(self, y_train):
        y_train = check_y(y_train)
        return self.update(y_train)

    def predict(self, fh):
        self._set_fh(fh)

        fh_abs = self.fh + self.cutoff
        if np.any(fh_abs < self.oh.index[0]):
            raise ValueError("cannot predict time point before oh")

        return self._predict(self.fh)

    def update(self, y_new):
        y_new = check_y(y_new)
        self.oh = y_new.combine_first(self.oh)
        self.cutoff = y_new.index[-1]
        return self

    def update_predict(self, y_test, cv):
        check_cv(cv)
        return self._predict_moving_cutoff(y_test, cv)

    def _predict(self, fh):
        is_oos = fh > 0
        is_ins = np.logical_not(is_oos)

        fh_oos = fh[is_oos]
        fh_ins = fh[is_ins]

        if np.all(is_oos):
            return self._predict_fixed_cutoff(fh_oos)
        elif np.all(is_ins):
            return self._predict_ins(fh_ins)
        else:
            y_ins = self._predict_ins(fh_ins)
            y_oos = self._predict_fixed_cutoff(fh_oos)
            return np.append(y_ins, y_oos)

    def _predict_ins(self, fh):
        assert all(fh <= 0)
        fh_abs = self.cutoff + fh

        # TODO fix this hack
        if hasattr(self, "_detached_oh_start") and np.any(fh_abs < self._detached_oh_start):
            # during moving cutoff prediction with in-sample data and in-sample forecasting
            # horizon, make sure to move first point to predict within the in-sample data,
            # rather than predicting a value before the passed in-sample data
            shift = np.min(fh_abs) - self._detached_oh_start
            fh_abs = fh_abs - shift
            cutoffs = fh_abs - 1
            # after adjusting the point to predict, it may happen that the required pre-sample
            # data is not available in the observation horizon, so we add it here
            if not np.all(np.isin(cutoffs, self.oh.index)):
                self.oh = self.oh.combine_first(pd.Series(np.full(len(cutoffs), np.nan), index=cutoffs))
        else:
            cutoffs = fh_abs - 1

        window_length = self.window_length
        fh = np.array([1])
        cv = ManualWindowSplitter(cutoffs, fh, window_length)
        y_pred = self._predict_moving_cutoff(self.oh, cv)
        return np.hstack(y_pred)

    def _predict_moving_cutoff(self, y, cv):
        # depending on cv and y, y may need some adjustment, so that
        # the first prediction is always the first point in y
        y = self._adjust_y(y, cv)

        fh = cv.fh
        r = []
        with self._detached_update():
            for i, (new_window, _) in enumerate(cv.split(y)):
                y_new = y.iloc[new_window]
                self.update(y_new)
                y_pred = self._predict(fh)
                r.append(y_pred)
        return r

    def _adjust_y(self, y, cv):
        window_length = cv.window_length
        cutoff = y.index[0] - 1
        start = cutoff - window_length

        # if start is before observation horizon,
        # add pre-sample padding
        if start < self.oh.index[0]:
            return self._pre_sample_pad(self.oh, start)

        # else prepend observation horizon
        else:
            return self.oh.iloc[start:].append(y)

    def _predict_fixed_cutoff(self, fh):
        assert all(fh > 0)
        return self._predict_last(fh)

    @contextmanager
    def _detached_update(self):
        cutoff = self.cutoff
        self._detached_oh_start = self.oh.index[0]
        try:
            yield
        finally:
            self.cutoff = cutoff
            self.oh = self.oh.loc[self._detached_oh_start:]

    def _predict_last(self, fh):
        # last strategy
        last_window = self._get_last_window()
        return np.repeat(last_window.values[-1], len(fh))

    @staticmethod
    def _pre_sample_pad(y, start):
        assert start < y.index[0]
        end = y.index[0]
        index = np.arange(start, end)
        pad = pd.Series(np.full(len(index), np.nan), index=index)
        y = pad.append(y)
        return y

    def _get_last_window(self):
        start = self.cutoff - self.window_length
        end = self.cutoff
        return self.oh.loc[start:end]

    def _set_fh(self, fh):
        check_fh(fh)
        self.fh = fh
        self.fh_abs = fh + self.cutoff


if __name__ == "__main__":
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split, SlidingWindowSplitter
    from sktime.utils.testing.forecasting import compute_expected_index_from_update_predict
    from pytest import raises


    def assert_equal_update_predict_index(y_pred, y_test, cv):
        a = y_pred.index.values
        b = compute_expected_index_from_update_predict(y_test, cv)
        np.testing.assert_array_equal(a, b)

    def test_predict_oos():
        f = Forecaster()
        f.fit(y_train)
        fh = np.array([2, 5])
        y_pred = f.predict(fh)
        np.testing.assert_array_equal(y_pred, np.repeat(y_train.iloc[-1], len(fh)))


    def test_predict_is():
        f = Forecaster()
        f.fit(y_train)
        fh = np.array([-2, -5])
        y_pred = f.predict(fh)
        np.testing.assert_array_equal(y_pred, y_train.iloc[fh + y_train.index[-1] - 1].values)


    def test_predict_is_negative_abs_fh():
        f = Forecaster()
        f.fit(y_train)
        fh = np.array([-len(y_train) - 1])
        with raises(ValueError):
            f.predict(fh)


    def test_update_predict_oos_y_test():
        f = Forecaster()
        f.fit(y_train)
        fh = np.array([2, 5])
        cv = SlidingWindowSplitter(fh=fh, window_length=3)
        y_pred = f.update_predict(y_test, cv)


    def test_update_predict_is_y_test():
        f = Forecaster()
        f.fit(y_train)
        fh = np.array([-2, -5])
        cv = SlidingWindowSplitter(fh=fh, window_length=3)
        y_pred = f.update_predict(y_test, cv)


    def test_update_predict_oos_y_train():
        f = Forecaster()
        f.fit(y_train)
        fh = np.array([2, 5])
        cv = SlidingWindowSplitter(fh=fh, window_length=3)
        y_pred = f.update_predict(y_train, cv)


    def test_update_predict_is_y_train():
        f = Forecaster()
        f.fit(y_train)
        fh = np.array([-2, -5])
        cv = SlidingWindowSplitter(fh=fh, window_length=3)
        y_pred = f.update_predict(y_train, cv)


    def test_check_tests():
        print("running tests")


    def run_tests():
        test_check_tests()
        test_predict_oos()
        test_predict_is()
        test_update_predict_oos_y_test()
        test_update_predict_oos_y_train()
        test_update_predict_is_y_test()
        test_update_predict_is_y_train()


    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=.3)
    run_tests()
