import numpy as np
from ovito.io import import_file
from ovito.modifiers import CalculateDisplacementsModifier
from ovito.pipeline import FileSource


DT_PS = 0.005
Q_LIST = np.array([0.3, 0.1])
# =========================================================================
# 2. 載入軌跡與設定外部參考
# =========================================================================
def calc_isf_multi_q(DUMP_FILE, OUTPUT_FILE):
    pipeline = import_file(DUMP_FILE)

    modifier = CalculateDisplacementsModifier()
    modifier.minimum_image_convention = True  
    pipeline.modifiers.append(modifier)

    num_frames = pipeline.source.num_frames

    # =========================================================================
    # 3. 逐幀計算 (Numpy 矩陣化加速)
    # =========================================================================
    # 第一行(t=0): 時間為0, ISF 全部為 1.0
    time_array = [0.0]
    isf_matrix = [np.ones(len(Q_LIST))] 

    for frame in range(num_frames):
        data = pipeline.compute(frame)
        timestep = data.attributes['Timestep']
        time_ps = timestep * DT_PS

        # 抓位移並扣質心飄移
        disp_vec = np.array(data.particles['Displacement'])
        com_drift = np.mean(disp_vec, axis=0)
        true_disp = np.linalg.norm(disp_vec - com_drift, axis=1) # shape: (N_atoms,)
        # 🌟 矩陣魔法：一次計算所有 q 值的 q*d
        # true_disp[:, np.newaxis] 變成 (N_atoms, 1)
        # Q_LIST 是 (N_q,)
        # 乘起來 qd_matrix 會變成 (N_atoms, N_q)
        qd_matrix = true_disp[:, np.newaxis] * Q_LIST

        # 對矩陣取 sinc，然後對原子軸 (axis=0) 取平均，得到每個 q 的 ISF
        isf_values = np.mean(np.sinc(qd_matrix / np.pi), axis=0)
        time_array.append(time_ps)
        isf_matrix.append(isf_values)

        if frame % 10 == 0 or frame == num_frames - 1:
            print(f"已處理 Frame {frame} (Time = {time_ps:.2f} ps)")

    # =========================================================================
    # 4. 儲存數據
    # =========================================================================
    # 把 time 和 ISF 矩陣併在一起
    final_data = np.column_stack((time_array, isf_matrix))

    # 動態產生 Header： Time(ps) q=0.05 q=0.075 ...
    header_str = "Time(ps) " + " ".join([f"q={q}" for q in Q_LIST])

    np.savetxt(OUTPUT_FILE, final_data, header=header_str, fmt="%.6f")
    print(f"\n計算完成！數據已儲存至 {OUTPUT_FILE}\n")

if __name__ == "__main__":
    directions = ["xy", "xz", "yz"]
    states = ["rx", "wo"]
    for i in states:
        for j in directions:
            state = i
            direction = j
            DUMP_FILE = f"Trajectories/Traj_{state}_merged.{direction}.lammpstrj.gz" # 你的軌跡檔
            OUTPUT_FILE = f"ISF/ISF_{state}.{direction}.dat"               # 輸出的矩陣檔

            calc_isf_multi_q(DUMP_FILE, OUTPUT_FILE)