import numpy as np

def merge_and_ave(input_base, output_file):
    xy_rx = np.loadtxt(f"{input_base}_xy.csv", skiprows=1, delimiter=',')
    xz_rx = np.loadtxt(f"{input_base}_xz.csv", skiprows=1, delimiter=',')
    yz_rx = np.loadtxt(f"{input_base}_yz.csv", skiprows=1, delimiter=',')

    # Merge and average the ACF curves
    time = xy_rx[:, 1] * 0.005  # Assuming the first column is time [ps], convert to ps
    merged_acf = (xy_rx[:, 7] + xz_rx[:, 7] + yz_rx[:, 7]) / 3
    output = np.column_stack((time, merged_acf))
    # Save the merged ACF curve to the output file
    np.savetxt(output_file, output, delimiter=' ', header='Merged BCF Curve', comments='# ')

if __name__ == "__main__":
    input_base = "BCF/topology_evolution"  # Base name for the input files
    output_file = "BCF/RXN_ave.dat"  # Output file for the averaged ACF curve
    merge_and_ave(input_base, output_file)
    print(f"Merged ACF curve saved to {output_file}")