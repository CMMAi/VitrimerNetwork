import numpy as np
import os
import glob

def merge_and_ave(input_base, output_file):
    # Get a list of all .npy files in the input directory
    xy_wo_files = sorted(glob.glob(f"{input_base}_wo_*.xy.dat"))
    yz_wo_files = sorted(glob.glob(f"{input_base}_wo_*.yz.dat"))
    xz_wo_files = sorted(glob.glob(f"{input_base}_wo_*.xz.dat"))
    xy_rx_files = sorted(glob.glob(f"{input_base}_rx_*.xy.dat"))
    yz_rx_files = sorted(glob.glob(f"{input_base}_rx_*.yz.dat"))
    xz_rx_files = sorted(glob.glob(f"{input_base}_rx_*.xz.dat"))
    xy_wo_arr = []
    yz_wo_arr = []
    xz_wo_arr = []
    xy_rx_arr = []
    yz_rx_arr = []
    xz_rx_arr = []

    for f in xy_wo_files:
        data = np.loadtxt(f, skiprows=2)
        xy_wo_arr.append(data)
    xy_wo_arr = np.concatenate(xy_wo_arr, axis=0)

    for f in yz_wo_files:
        data = np.loadtxt(f, skiprows=2)
        yz_wo_arr.append(data)
    yz_wo_arr = np.concatenate(yz_wo_arr, axis=0)

    for f in xz_wo_files:
        data = np.loadtxt(f, skiprows=2)
        xz_wo_arr.append(data)
    xz_wo_arr = np.concatenate(xz_wo_arr, axis=0)

    for f in xy_rx_files:
        data = np.loadtxt(f, skiprows=2)
        xy_rx_arr.append(data)
    xy_rx_arr = np.concatenate(xy_rx_arr, axis=0)

    for f in yz_rx_files:
        data = np.loadtxt(f, skiprows=2)
        yz_rx_arr.append(data)
    yz_rx_arr = np.concatenate(yz_rx_arr, axis=0)

    for f in xz_rx_files:
        data = np.loadtxt(f, skiprows=2)
        xz_rx_arr.append(data)
    xz_rx_arr = np.concatenate(xz_rx_arr, axis=0)

    time = xy_wo_arr[:, 0]
    stress_xy_wo = xy_wo_arr[:, 2]
    stress_yz_wo = yz_wo_arr[:, 2]
    stress_xz_wo = xz_wo_arr[:, 2]

    stress_xy_rx = xy_rx_arr[:, 2]
    stress_yz_rx = yz_rx_arr[:, 2]
    stress_xz_rx = xz_rx_arr[:, 2]

    reduced_xy_wo = stress_xy_wo / stress_xy_wo[0]
    reduced_yz_wo = stress_yz_wo / stress_yz_wo[0]
    reduced_xz_wo = stress_xz_wo / stress_xz_wo[0]
    
    reduced_xy_rx = stress_xy_rx / stress_xy_rx[0]
    reduced_yz_rx = stress_yz_rx / stress_yz_rx[0]
    reduced_xz_rx = stress_xz_rx / stress_xz_rx[0]

    reduced_stress_wo = (reduced_xy_wo + reduced_yz_wo + reduced_xz_wo) / 3
    reduced_stress_rx = (reduced_xy_rx + reduced_yz_rx + reduced_xz_rx) / 3

    time = (time - time[0]) * 0.005 # ps

    arr_wo = np.column_stack((time, reduced_stress_wo))
    arr_rx = np.column_stack((time, reduced_stress_rx))

    np.savetxt(f"{output_file}_wo.dat", arr_wo, fmt="%f", header="Time(ps) Reduced_Stress")
    np.savetxt(f"{output_file}_rx.dat", arr_rx, fmt="%f", header="Time(ps) Reduced_Stress")


if __name__ == "__main__":
    input_base = "CG_DGEBA_SeA_SRT_450"
    output_file = "stress_ave"
    merge_and_ave(input_base, output_file)