""" test parquet compat """

import pytest
pyarrow = pytest.importorskip('pyarrow')

import numpy as np
import pandas as pd
from pandas.io.parquet import to_parquet, read_parquet

from pyarrow.error import ArrowException
from pandas.util.testing import assert_frame_equal, ensure_clean


class TestParquet(object):

    def check_error_on_write(self, df, exc):
        # check that we are raising the exception
        # on writing

        with pytest.raises(exc):
            with ensure_clean() as path:
                to_parquet(df, path)

    def check_round_trip(self, df, expected=None, **kwargs):

        with ensure_clean() as path:
            to_parquet(df, path, **kwargs)
            result = read_parquet(path)

            if expected is None:
                expected = df
            assert_frame_equal(result, expected)

    def test_error(self):

        for obj in [pd.Series([1, 2, 3]), 1, 'foo', pd.Timestamp('20130101'),
                    np.array([1, 2, 3])]:
            self.check_error_on_write(obj, ValueError)

    def test_basic(self):

        df = pd.DataFrame({'string': list('abc'),
                           'int': list(range(1, 4)),
                           'uint': np.arange(3, 6).astype('u1'),
                           'float': np.arange(4.0, 7.0, dtype='float64'),
                           'bool': [True, False, True],
                           })

        self.check_round_trip(df)

    def test_duplicate_columns(self):

        # not currently able to handle duplicate columns
        df = pd.DataFrame(np.arange(12).reshape(4, 3),
                          columns=list('aaa')).copy()
        self.check_error_on_write(df, ValueError)

    def test_stringify_columns(self):

        df = pd.DataFrame(np.arange(12).reshape(4, 3)).copy()
        self.check_error_on_write(df, ValueError)

    def test_unsupported(self):

        # period
        df = pd.DataFrame({'a': pd.period_range('2013', freq='M', periods=3)})
        self.check_error_on_write(df, ArrowException)

        # categorical
        df = pd.DataFrame({'a': pd.Categorical(list('abc'))})
        self.check_error_on_write(df, ArrowException)

        # date_range
        df = pd.DataFrame({'a': pd.date_range('20130101', periods=3)})
        self.check_error_on_write(df, ArrowException)

        # date_range w/tz
        df = pd.DataFrame({'a': pd.date_range('20130101', periods=3,
                                              tz='US/Eastern')})
        self.check_error_on_write(df, ArrowException)

    def test_mixed(self):
        # mixed python objects are returned as None ATM
        df = pd.DataFrame({'a': ['a', 1, 2.0]})
        expected = pd.DataFrame({'a': ['a', None, None]})

        self.check_round_trip(df, expected)

    def test_write_with_index(self):

        df = pd.DataFrame({'A': [1, 2, 3]})
        self.check_round_trip(df)

        # non-default index
        for index in [[2, 3, 4],
                      pd.date_range('20130101', periods=3),
                      list('abc'),
                      [1, 3, 4],
                      pd.MultiIndex.from_tuples([('a', 1), ('a', 2),
                                                 ('b', 1)]),
                      ]:

            df.index = index
            self.check_error_on_write(df, ValueError)

        # index with meta-data
        df.index = [0, 1, 2]
        df.index.name = 'foo'
        self.check_error_on_write(df, ValueError)

        # column multi-index
        df.index = [0, 1, 2]
        df.columns = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1)]),
        self.check_error_on_write(df, ValueError)

    @pytest.mark.parametrize('compression', [None, 'gzip', 'snappy'])
    def test_compression(self, compression):

        df = pd.DataFrame({'A': [1, 2, 3]})
        self.check_round_trip(df, compression=compression)
