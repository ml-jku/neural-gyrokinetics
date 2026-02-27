import pytest
from neugk.dataset.backend import KvikIOBackend, H5Backend


def test_kvikio_backend_path_stripping():
    # Mocking since we don't need a real backend object for static-like methods
    # But KvikIOBackend requires cupy/kvikio imports which might fail in non-GPU env
    # Let's see if we can at least test _strip_h5

    backend = KvikIOBackend()

    # Test .h5 stripping
    assert backend._strip_h5("path/to/data.h5") == "path/to/data"
    assert backend._strip_h5("path/to/data") == "path/to/data"
    assert backend._strip_h5("/tmp/test.h5/") == "/tmp/test"

    # Test format_path
    # format_path(self, path, spatial_ifft, split_into_bands=None, real_potens=False)
    # KvikIO logic adds tags to the folder name
    path = "base/path"
    f1 = backend.format_path(path, spatial_ifft=True)
    assert "_ifft" in f1

    f2 = backend.format_path(path, spatial_ifft=True, split_into_bands=2)
    assert "_2bands" in f2

    f3 = backend.format_path(path, spatial_ifft=True, real_potens=True)
    assert "_realpotens" in f3


def test_h5_backend_format_path():
    backend = H5Backend()
    path = "base/path.h5"

    # H5Backend adds tags before the extension
    # default real_potens is True, so it adds _ifft_realpotens.h5
    f1 = backend.format_path(path, spatial_ifft=True)
    assert "base/path_ifft_realpotens.h5" == f1

    # explicitly False
    f1_no_rp = backend.format_path(path, spatial_ifft=True, real_potens=False)
    assert "base/path_ifft.h5" == f1_no_rp

    f2 = backend.format_path(path, spatial_ifft=True, split_into_bands=4)
    assert "base/path_ifft_separate_zf_4bands.h5" == f2
