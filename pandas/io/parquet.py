""" parquet compat """

from pandas import DataFrame, RangeIndex, Int64Index
from pandas.compat import range


def _try_import():
    # since pandas is a dependency of parquet
    # we need to import on first use

    try:
        import pyarrow
    except ImportError:

        # give a nice error message
        raise ImportError("the pyarrow library is not installed\n"
                          "you can install via conda\n"
                          "conda install pyarrow -c conda-forge\n"
                          "or via pip\n"
                          "pip install pyarrow\n")

    return pyarrow


def to_parquet(df, path, compression=None):
    """
    Write a DataFrame to the pyarrow

    Parameters
    ----------
    df : DataFrame
    path : string
        File path
    compression : str, optional
        compression method, includes {'gzip', 'snappy'}
    """
    if not isinstance(df, DataFrame):
        raise ValueError("pyarrow only support IO with DataFrames")

    pyarrow = _try_import()
    valid_types = {'string', 'unicode'}

    # validate index
    # --------------

    # validate that we have only a default index
    # raise on anything else as we don't serialize the index

    if not isinstance(df.index, Int64Index):
        raise ValueError("feather does not serializing {} "
                         "for the index; you can .reset_index()"
                         "to make the index into column(s)".format(
                             type(df.index)))

    if not df.index.equals(RangeIndex.from_range(range(len(df)))):
        raise ValueError("feather does not serializing a non-default index "
                         "for the index; you can .reset_index()"
                         "to make the index into column(s)")

    if df.index.name is not None:
        raise ValueError("feather does not serialize index meta-data on a "
                         "default index")

    # validate columns
    # ----------------

    # must have value column names (strings only)
    if df.columns.inferred_type not in valid_types:
        raise ValueError("feather must have string column names")

    from pyarrow import parquet as pq

    table = pyarrow.Table.from_pandas(df)
    pq.write_table(table, path, compression=compression)


def read_parquet(path):
    """
    Load a paruquet object from the file path

    .. versionadded 0.20.0

    Parameters
    ----------
    path : string
        File path

    Returns
    -------
    type of object stored in file

    """

    pyarrow = _try_import()
    return pyarrow.parquet.read_table(path).to_pandas()
