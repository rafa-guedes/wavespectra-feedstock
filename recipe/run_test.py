"""Smoke tests for the wavespectra conda package.

`wavespectra/__init__.py` catches ImportError when importing the xarray
accessors, and `_import_functions` does the same for every `read_*` reader,
downgrading both to warnings so the package stays importable. `import
wavespectra` therefore succeeds even when a runtime dependency is missing,
which means `test: imports:` plus `pip check` cannot detect an over-trimmed
`run:` section.

These checks import the accessor and reader modules directly, so a missing
dependency raises ImportError instead of being swallowed, and then exercise the
accessor and the compiled `specpart` extension end to end.
"""

import numpy as np


def test_accessors_import_directly():
    """Import the accessor modules bypassing __init__'s try/except."""
    from wavespectra.specarray import SpecArray
    from wavespectra.specdataset import SpecDataset

    assert SpecArray is not None
    assert SpecDataset is not None


def test_readers_import_directly():
    """Readers we require must import without their ImportError swallowed.

    Imported by module path rather than via `hasattr(wavespectra, ...)`, since
    __init__ simply omits the attribute when the reader fails to import.
    Only readers whose dependencies are in this package's `run:` section are
    listed; readers needing optional extras (netCDF4, h5netcdf) are excluded.
    """
    from wavespectra.input.ww3 import read_ww3
    from wavespectra.input.swan import read_swan
    from wavespectra.input.era5 import read_era5
    from wavespectra.input.ndbc_ascii import read_ndbc_ascii

    for func in (read_ww3, read_swan, read_era5, read_ndbc_ascii):
        assert callable(func)


def test_accessor_is_registered():
    """The `.spec` accessor is attached to xarray objects."""
    import xarray as xr
    import wavespectra  # noqa: F401

    assert hasattr(xr.DataArray, "spec")
    assert hasattr(xr.Dataset, "spec")


def _spectrum():
    from wavespectra import construct

    return construct.construct_partition(
        freq_name="jonswap",
        freq_kwargs={"freq": np.arange(0.04, 0.4, 0.01), "fp": 0.1, "hs": 2.0},
        dir_name="cartwright",
        dir_kwargs={"dir": np.arange(0, 360, 10), "dm": 90.0, "dspr": 25.0},
    )


def test_construct_and_stats():
    """Build a JONSWAP spectrum and recover its Hs and Tp through `.spec`."""
    dset = _spectrum()
    assert abs(float(dset.spec.hs()) - 2.0) < 0.01
    assert abs(float(dset.spec.tp()) - 10.0) < 0.5


def test_partition_extension():
    """The compiled specpart extension imports and runs."""
    from wavespectra.partition import specpart  # noqa: F401

    parts = _spectrum().spec.partition.ptm1(
        wspd=10.0, wdir=90.0, dpt=100.0, agefac=1.7, wscut=0.3333
    )
    assert "part" in parts.dims


def test_cli_entry_point():
    """The console script imports; covers the `click` run dependency."""
    from wavespectra.cli import main

    assert callable(main)


def test_plot():
    """Plotting works; covers the `matplotlib-base` run dependency."""
    import matplotlib

    matplotlib.use("Agg")
    assert _spectrum().spec.plot() is not None


if __name__ == "__main__":
    test_accessors_import_directly()
    test_readers_import_directly()
    test_accessor_is_registered()
    test_construct_and_stats()
    test_partition_extension()
    test_cli_entry_point()
    test_plot()
    print("all wavespectra smoke tests passed")
