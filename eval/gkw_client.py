from typing import Dict, List
import zmq
import time
import numpy as np
import os

PORT = 13377


def modify_fds_dat(path):
    with open(path, "r") as infile:
        content = infile.read()
        content = content.replace("DTIM    =  2.000000000000000E-002", "DTIM    =  0.0")
        content = content.replace(
            "NT_REMAIN       =           0", "NT_REMAIN       =           1"
        )
        content = content.replace("TIME    =   192.753733197446     ", "TIME    =   0")

    with open(path, "w") as outfile:
        outfile.write(content)


def modify_input_dat(path):
    with open(path, "r") as infile:
        content = infile.read()
        content = content.replace("READ_FILE  = .false.", "READ_FILE  = .true.")
        content = content.replace("DTIM   = 0.02", "DTIM   = 0.0")
        content = content.replace("out3d_interval = 3", "out3d_interval = 1")
        content = content.replace("keep_dumps = .true.", "! keep_dumps = .true.")

    with open(path, "w") as outfile:
        outfile.write(content)


def dump_rollout(
    rollout: Dict[str, np.ndarray],
    kfile_dest_path: str,
    src_config_path: str,
    dump_ratio: int = 20,
) -> List[str]:
    kfiles = []
    # subsample dumps TODO (only dumped at 20x because of n_eval_steps??)
    rollout = {k: v for k, v in rollout.items() if (k % dump_ratio) == 0}
    for idx, xt in rollout.items():
        kfile = f"K{str(int(idx) + 1).zfill(5)}"
        dirtarget = os.path.join(kfile_dest_path, kfile)
        os.makedirs(dirtarget, exist_ok=True)
        os.system(f"cp {src_config_path}/FDS.dat {dirtarget}")
        os.system(f"cp {src_config_path}/input.dat {dirtarget}")
        modify_fds_dat(f"{dirtarget}/FDS.dat")
        modify_input_dat(f"{dirtarget}/input.dat")
        ftarget = os.path.join(dirtarget, "FDS")
        # convert x to fourier
        xt = np.ascontiguousarray(np.moveaxis(xt, 0, -1))
        xt = xt.view(dtype=np.complex64)
        # shift freqs to correct range
        xt = np.fft.fftn(xt, axes=(3, 4))
        xt = np.fft.ifftshift(xt, axes=(3, 4))
        xt = np.stack([xt.real, xt.imag]).squeeze()
        with open(ftarget, "wb") as f:
            kfiles.append(kfile)
            # dump to file
            f.write(xt.astype("float64").reshape(-1, order="F"))
        os.system(f"chmod -R 777 {dirtarget}/*")

    return kfiles


def cleanup_kfiles(path: str, kfiles: List[str]):
    # TODO
    pass


def request_gkw_sim(
    dump_path: str, kfiles: List[str], run_id: str, terminate_on_flux: bool = True
):
    # TODO with pollers to check heartbeat of server
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://pike.bioinf.jku.at:{PORT}")

    socket.send_json(
        {
            "path": dump_path,
            "kfiles": kfiles,
            "terminate_on_flux": terminate_on_flux,
            "run_id": run_id,  # run ID here
            "timestamp": time.time(),
        }
    )
    results = socket.recv_json()

    sgrid = np.loadtxt(f"{dump_path}/{kfiles[0]}/sgrid")
    xphi = np.loadtxt(f"{dump_path}/{kfiles[0]}/xphi")
    fluxes = {k.replace(f"{dump_path}/", ""): v for k, v in results["fluxes"].items()}
    potentials = {}
    for kf in kfiles:
        a = np.loadtxt(f"{dump_path}/{kf}/Poten00000501")
        ns = sgrid.shape[1] if len(sgrid.shape) > 1 else sgrid.shape[0]
        nx, ny = xphi.shape[1], xphi.shape[0]
        potentials[kf] = np.reshape(a, (nx, ns, ny), order="F")
    cleanup_kfiles(dump_path, kfiles)
    return fluxes, potentials
