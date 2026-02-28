from typing import Optional, Sequence, Any, Dict

import os
import re
from abc import ABC, abstractmethod

import h5py
import contextlib
import pickle

import numpy as np
import torch


def read_cupy_bin(file: str, shape: tuple, rank: int = 0, use_kvikio: bool = True):
    if use_kvikio:
        import cupy as cp
        import kvikio

        n_elements = np.prod(shape)
        with cp.cuda.Device(rank):
            gpu_array = cp.empty(n_elements, dtype=cp.float32)
            with kvikio.CuFile(file, "r") as f:
                f.read(gpu_array)
        return torch.from_dlpack(gpu_array.reshape(shape))

    else:
        cpu_array = np.fromfile(file, dtype=np.float32)
        return torch.from_numpy(cpu_array.reshape(shape))


class DataBackend(ABC):
    def __init__(self, rank: int = 0):
        self.rank = rank

    @abstractmethod
    def is_valid(self, path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def format_path(
        self,
        path: str,
        spatial_ifft: bool,
        split_into_bands: Optional[int] = None,
        real_potens: bool = True,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_metadata(
        self, path: str, input_fields: Sequence[str] = ["df"]
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def open(self, path: str):
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def create(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def read_df(
        self,
        f: Any,
        timestamp: str,
        shape: Sequence[int],
        active_keys: Optional[Sequence[str]] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def read_phi(self, f: Any, timestamp: str, shape: Sequence[int]):
        raise NotImplementedError

    @abstractmethod
    def write_metadata(self, f: Any, metadata: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def write_df(self, f: Any, timestamp: str, df: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def write_phi(self, f: Any, timestamp: str, phi: np.ndarray):
        raise NotImplementedError


class H5Backend(DataBackend):
    def is_valid(self, path: str) -> bool:
        return os.path.isfile(path)

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def format_path(
        self,
        path: str,
        spatial_ifft: bool,
        split_into_bands: Optional[int] = None,
        real_potens: bool = True,
    ) -> str:
        if spatial_ifft:
            n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
            tag = (
                f"_ifft_separate_zf{n_bands_tag}.h5"
                if split_into_bands
                else ("_ifft_realpotens.h5" if real_potens else "_ifft.h5")
            )
            path = path if tag in path else re.sub(r"\.h5$", tag, path)
        elif not path.endswith(".h5"):
            path += ".h5"
        return path

    def read_metadata(
        self, path: str, input_fields: Sequence[str] = ["df"]
    ) -> Dict[str, Any]:
        meta = {}
        with h5py.File(path, "r", swmr=True) as f:
            meta["timesteps"] = f["metadata/timesteps"][:]
            meta["fluxes"] = f["metadata/fluxes"][:]
            meta["ion_temp_grad"] = f["metadata/ion_temp_grad"][:]
            meta["density_grad"] = f["metadata/density_grad"][:]
            meta["s_hat"] = f["metadata/s_hat"][:]
            meta["q"] = f["metadata/q"][:]
            meta["resolution"] = f["metadata/resolution"][:]
            meta["geometry"] = {k: np.array(v[()]) for k, v in f["geometry"].items()}
            if "adiabatic" not in meta["geometry"]:
                meta["geometry"]["adiabatic"] = np.array(1.0, dtype=np.float64)
            if "de" not in meta["geometry"]:
                meta["geometry"]["de"] = np.array(1.0, dtype=np.float64)

            for k in input_fields:
                if f"metadata/{k}_mean" in f:
                    meta[f"{k}_mean"] = f[f"metadata/{k}_mean"][:]
                    meta[f"{k}_std"] = f[f"metadata/{k}_std"][:]
                    meta[f"{k}_min"] = f[f"metadata/{k}_min"][:]
                    meta[f"{k}_max"] = f[f"metadata/{k}_max"][:]
        return meta

    @contextlib.contextmanager
    def open(self, path: str):
        f = h5py.File(path, "r", swmr=True)
        try:
            yield f
        finally:
            f.close()

    @contextlib.contextmanager
    def create(self, path: str):
        write_mode = "a" if os.path.exists(path) else "w"
        f = h5py.File(path, write_mode)
        try:
            if "data" not in f.keys():
                f.create_group("data")
            yield f
        finally:
            f.close()

    def read_df(
        self,
        f,
        timestamp: str,
        shape: Sequence[int],
        active_keys: Optional[Sequence[str]] = None,
    ):
        _ = shape, self.rank
        k = f[f"data/timestep_{timestamp}"][:]
        if all(active_keys == np.array([0, 1])):
            return k
        return k[active_keys]

    def read_phi(self, f, timestamp: str, shape: Sequence[int]):
        _ = shape, self.rank
        return f[f"data/poten_{timestamp}"][:]

    def write_metadata(self, f, metadata: Dict[str, Any]):
        if "metadata" not in f.keys():
            meta_grp = f.create_group("metadata")
        else:
            meta_grp = f["metadata"]

        if "geometry" not in f.keys():
            geom_grp = f.create_group("geometry")
        else:
            geom_grp = f["geometry"]

        for k, v in metadata.items():
            if k == "geometry":
                for gk, gv in v.items():
                    if gk not in geom_grp.keys():
                        geom_grp.create_dataset(gk, data=gv)
            else:
                if k not in meta_grp.keys():
                    meta_grp.create_dataset(k, data=v)

    def write_df(self, f, timestamp: str, df: np.ndarray):
        k_name = f"timestep_{timestamp}"
        if k_name not in f["data"].keys():
            f["data"].create_dataset(k_name, data=df)

    def write_phi(self, f, timestamp: str, phi: np.ndarray):
        p_name = f"poten_{timestamp}"
        if p_name not in f["data"].keys():
            f["data"].create_dataset(p_name, data=phi)


class KvikIOBackend(DataBackend):
    def __init__(self, rank: int = 0, use_kvikio: bool = True):
        super().__init__(rank)

        self.use_kvikio = use_kvikio

    def _strip_h5(self, path: str) -> str:
        return path.removesuffix("/").removesuffix(".h5")

    def is_valid(self, path: str) -> bool:
        path = self._strip_h5(path)
        return os.path.isdir(path)

    def exists(self, path: str) -> bool:
        path = self._strip_h5(path)
        return os.path.exists(os.path.join(path, "metadata.pkl"))

    def format_path(
        self,
        path: str,
        spatial_ifft: bool,
        split_into_bands: Optional[int] = None,
        real_potens: bool = True,
    ) -> str:
        path = self._strip_h5(path)
        if spatial_ifft:
            n_bands_tag = f"_{split_into_bands}bands" if split_into_bands else ""
            tag = (
                f"_ifft_separate_zf{n_bands_tag}_realpotens"
                if split_into_bands
                else ("_ifft_realpotens" if real_potens else "_ifft")
            )
            path = path if tag in path else path + tag
        return path

    def read_metadata(
        self, path: str, input_fields: Sequence[str] = ["df"]
    ) -> Dict[str, Any]:
        _ = input_fields
        path = self._strip_h5(path)
        with open(os.path.join(path, "metadata.pkl"), "rb") as mf:
            meta = pickle.load(mf)
            if "geometry" in meta:
                if "adiabatic" not in meta["geometry"]:
                    meta["geometry"]["adiabatic"] = np.array(1.0, dtype=np.float64)
                if "de" not in meta["geometry"]:
                    meta["geometry"]["de"] = np.array(1.0, dtype=np.float64)
            return meta

    @contextlib.contextmanager
    def open(self, path: str):
        yield path

    @contextlib.contextmanager
    def create(self, path: str):
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        yield path

    def read_df(
        self,
        f_dir: str,
        timestamp: str,
        shape: Sequence[int],
        active_keys: Optional[Sequence[str]] = None,
    ):
        filepath = os.path.join(f_dir, "data", f"timestep_{timestamp}.bin")
        k = read_cupy_bin(filepath, shape, self.rank, self.use_kvikio)

        if all(active_keys == np.array([0, 1])):
            return k
        return k[active_keys]

    def read_phi(self, f_dir: str, timestamp: str, shape: Sequence[int]):
        filepath = os.path.join(f_dir, "data", f"poten_{timestamp}.bin")
        return read_cupy_bin(filepath, shape, self.rank, self.use_kvikio)

    def write_df(self, f_dir: str, timestamp: str, df: np.ndarray):
        data_dir = os.path.join(f_dir, "data")
        k_name = os.path.join(data_dir, f"timestep_{timestamp}.bin")
        if not os.path.exists(k_name):
            np.ascontiguousarray(df).tofile(k_name)

    def write_phi(self, f_dir: str, timestamp: str, phi: np.ndarray):
        data_dir = os.path.join(f_dir, "data")
        p_name = os.path.join(data_dir, f"poten_{timestamp}.bin")
        if not os.path.exists(p_name):
            np.ascontiguousarray(phi).tofile(p_name)

    def write_metadata(self, f_dir: str, metadata: Dict[str, Any]):
        meta_path = os.path.join(f_dir, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)
