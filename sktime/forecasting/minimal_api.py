#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = []

from contextlib import contextmanager

import numpy as np
import pandas as pd
from sktime.forecasting.model_selection import ManualWindowSplitter
from sktime.utils.validation.forecasting import check_fh, check_y, check_cv


class ForecastingHorizon:

    def __init__(self, values, relative=True):
        self._values = check_fh(values)

        if not isinstance(relative, bool):
            raise ValueError(f"relative must be a boolean, but found: {type(relative)}")
        self._relative = relative

    @property
    def values(self):
        return self._values

    @property
    def relative(self):
        return self._relative

    @property
    def in_sample_values(self):
        return self.values[self.values <= 0]

    @property
    def out_of_sample_values(self):
        return self.values[self.values > 0]

    def get_absolute_values(self, cutoff=None):
        if self.relative:
            return self.values + cutoff
        else:
            return self.values

    def get_relative_values(self, cutoff=None):
        if self.relative:
            return self.values
        else:
            return self.values - cutoff


class Forecaster:

    def __init__(self):
        self.oh = pd.Series([])
        self.cutoff = None
        self.window_length = 1
        self.fh = None

    def fit(self, y_train):
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
        self.update(y_test)
        cv = check_cv(cv)
        return self._predict_moving_cutoff(y_test, cv)

    def _predict(self, fh):
        is_oos = fh > 0
        is_ins = np.logical_not(is_oos)

        fh_oos = fh[is_oos]
        fh_ins = fh[is_ins]

        if all(is_oos):
            return self._predict_fixed_cutoff(fh_oos)
        elif all(is_ins):
            return self._predict_ins(fh_ins)
        else:
            y_ins = self._predict_ins(fh_ins)
            y_oos = self._predict_fixed_cutoff(fh_oos)
            return np.append(y_ins, y_oos)

    def _predict_ins(self, fh):
        assert all(fh <= 0)
        fh_abs = self.cutoff + fh
        cutoffs = fh_abs - 1
        window_length = self.window_length
        start = np.min(cutoffs) - window_length

        # if start is before observation horizon,
        # add pre-sample padding
        y_train = self.oh
        if start < self.oh.index[0]:
            y_train = self._pre_sample_pad(y_train, start)

        fh = np.array([1])
        cv = ManualWindowSplitter(cutoffs, fh, self.window_length)
        y_pred = self._predict_moving_cutoff(y_train, cv)
        return np.hstack(y_pred)

    def _predict_moving_cutoff(self, y, cv):
        # depending on cv and y, y may need some adjustment, so that
        # the first prediction is always the first point in y
        fh = cv.fh
        r = []
        with self._detached_update():
            for i, (new_window, _) in enumerate(cv.split(y)):
                y_new = y.iloc[new_window]
                self.update(y_new)
                y_pred = self._predict(fh)
                r.append(y_pred)
        return r

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
        last_window = self._get_last_window()
        return np.repeat(last_window[-1], len(fh))

    @staticmethod
    def _pre_sample_pad(y, start):
        assert start < y.index[0]
        end = y.index[0]
        index = np.arange(start, end)
        pad = pd.Series(np.full(len(index), np.nan), index=index)
        return pad.append(y)

    def _get_last_window(self):
        start = self.cutoff - self.window_length
        end = self.cutoff
        return self.oh.loc[start:end].values

    def _set_fh(self, fh):
        self.fh = check_fh(fh)


if __name__ == "__main__":
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split, SlidingWindowSplitter

    def test_check_tests():
        print("running tests")

    def test():
        f = Forecaster()
        f.fit(y_train)
        cv = SlidingWindowSplitter(fh=[0, 1], window_length=1, step_length=5)
        y_pred = f.update_predict(y_test, cv)

    def run_tests():
        test_check_tests()
        test()

    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=.3)
    run_tests()
