import os
import pickle
import subprocess
import argparse

DUMP_ROOT = "/system/user/publicdata/gyrokinetics/dumps"
DUMP_DIR = "tmp_swinv2_att_onestep"
GKW_HOME = "/system/user/galletti/git/gkw/"
MPIRUN = "/usr/lib64/openmpi/bin/mpirun"
CMD = f"{MPIRUN} -np 64 {GKW_HOME}/run/gkw_pike-DP_0.4-b1-1250-gd3c02d8a-dirty.x"


def main(dump_dir):
    fluxes = {}
    files = os.listdir(dump_dir)
    files = [f for f in files if f.startswith("K") and f != "fluxes_dict"]
    print(files)
    for k_dir in files:
        gkw_in_dir = f"{dump_dir}/{k_dir}"
        try:
            os.chdir(gkw_in_dir)
            print(os.getcwd(), end="")
            process = subprocess.Popen(
                CMD, shell=True, stdout=subprocess.PIPE, text=True
            )
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                if "Energy flux" in output:
                    flux = float(
                        output.strip().replace(" ", "").replace("Energyflux:", "")
                    )
                    print(f"\tflux: {flux}")

                    fluxes[k_dir] = flux
                    process.terminate()
                    break
        except Exception as e:
            print(f"Error processing {k_dir}: {e}")

    print("\n", fluxes)
    with open(f"{dump_dir}/fluxes_dict", "wb") as f:
        pickle.dump(fluxes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GKW simulation and extract energy flux values."
    )
    parser.add_argument(
        "dump_dir",
        type=str,
        help="Path to the directory containing simulation data.",
        default=DUMP_DIR,
    )
    args = parser.parse_args()
    dump_dir = f"{DUMP_ROOT}/{args.dump_dir}"
    main(dump_dir)
