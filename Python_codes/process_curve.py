import numpy as np
import glob

def merge_and_ave(input_base, output_file):
    # Get a list of all .npy files in the input directory
    xy_wo_files = sorted(glob.glob(f"{input_base}_wo.xy.dat"))
    yz_wo_files = sorted(glob.glob(f"{input_base}_wo.yz.dat"))
    xz_wo_files = sorted(glob.glob(f"{input_base}_wo.xz.dat"))
    xy_rx_files = sorted(glob.glob(f"{input_base}_rx.xy.dat"))
    yz_rx_files = sorted(glob.glob(f"{input_base}_rx.yz.dat"))
    xz_rx_files = sorted(glob.glob(f"{input_base}_rx.xz.dat"))

    xy_wo_arr = np.loadtxt(xy_wo_files[0], skiprows=2)
    yz_wo_arr = np.loadtxt(yz_wo_files[0], skiprows=2)
    xz_wo_arr = np.loadtxt(xz_wo_files[0], skiprows=2)
    xy_rx_arr = np.loadtxt(xy_rx_files[0], skiprows=2)
    yz_rx_arr = np.loadtxt(yz_rx_files[0], skiprows=2)
    xz_rx_arr = np.loadtxt(xz_rx_files[0], skiprows=2)

    time = xy_wo_arr[:, 0]
    P2_xy_wo = xy_wo_arr[:, 1]
    P2_yz_wo = yz_wo_arr[:, 1]
    P2_xz_wo = xz_wo_arr[:, 1]

    P2_xy_rx = xy_rx_arr[:, 1]
    P2_yz_rx = yz_rx_arr[:, 1]
    P2_xz_rx = xz_rx_arr[:, 1]

    reduced_P2_wo = (P2_xy_wo + P2_yz_wo + P2_xz_wo) / 3
    reduced_P2_rx = (P2_xy_rx + P2_yz_rx + P2_xz_rx) / 3

    arr_wo = np.column_stack((time, reduced_P2_wo))
    arr_rx = np.column_stack((time, reduced_P2_rx))

    np.savetxt(f"{output_file}_wo.dat", arr_wo, fmt="%f", header="Time(ps) Reduced_Int_long_MSD")
    np.savetxt(f"{output_file}_rx.dat", arr_rx, fmt="%f", header="Time(ps) Reduced_Int_long_MSD")


if __name__ == "__main__":
    input_base = "Rouse_dyn/Int_long"
    output_file = "Rouse_dyn/Int_long_ave"
    merge_and_ave(input_base, output_file)