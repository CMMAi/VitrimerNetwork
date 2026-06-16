import numpy as np

def merge_and_ave(input_base, output_file):
    xy_rx = np.loadtxt(f"{input_base}.xy.dat", skiprows=2)
    xz_rx = np.loadtxt(f"{input_base}.xz.dat", skiprows=2)
    yz_rx = np.loadtxt(f"{input_base}.yz.dat", skiprows=2)

    # Merge and average the ACF curves
    time = xy_rx[:, 0]  # Assuming the first column is time
    if xy_rx.shape[1] == 3:
        merged_acf1 = (xy_rx[:, 1] + xz_rx[:, 1] + yz_rx[:, 1]) / 3
        merged_acf2 = (xy_rx[:, 2] + xz_rx[:, 2] + yz_rx[:, 2]) / 3
        merged_acf = np.column_stack((merged_acf1, merged_acf2))
    else:
        merged_acf = (xy_rx[:, 1] + xz_rx[:, 1] + yz_rx[:, 1]) / 3
    output = np.column_stack((time, merged_acf))
    # Save the merged ACF curve to the output file
    np.savetxt(output_file, output, delimiter=' ', header='Merged Curve', comments='# ')

if __name__ == "__main__":
    input_base = "ISF/ISF_rx"  # Base name for the input files
    output_file = "ISF/ISF_rx_ave.dat"  # Output file for the averaged ACF curve
    merge_and_ave(input_base, output_file)
    print(f"Merged curve saved to {output_file}")