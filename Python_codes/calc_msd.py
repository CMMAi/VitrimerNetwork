from ovito.io import import_file
from ovito.modifiers import CalculateDisplacementsModifier
from ovito.pipeline import FileSource
import numpy as np

# =========================================================================
# 1. 使用者設定區
# =========================================================================
FILE_Prefix = "trajectory_500_SRT_rx"

REF_FILE = f"{FILE_Prefix}.lammpstrj.gz"

# 包含 Step 10 之後的軌跡檔案 (Log dump)
DUMP_FILE = f"{FILE_Prefix}_log.lammpstrj.gz" 

OUTPUT_FILE = f"{FILE_Prefix}_isf.dat"                    
Q_VALUE = 0.15                                    
DT_PS = 0.005     

# =========================================================================
# 2. 載入軌跡與設定 Modifier
# =========================================================================
print(f"載入檔案: {DUMP_FILE} ...")
pipeline = import_file(DUMP_FILE)

# 加入「計算位移」的 Modifier
# 預設會自動以第 0 幀 (Frame 0) 作為參考構型
print(f"正在將 {REF_FILE} 設為 t=0 的初始參考基準...")
ref_source = FileSource()
ref_source.load(REF_FILE)
disp_mod = CalculateDisplacementsModifier()
disp_mod.reference = ref_source 
disp_mod.minimum_image_convention = False  
pipeline.modifiers.append(disp_mod)

# =========================================================================
# 3. 迴圈計算每一幀的 MSD
# =========================================================================
num_frames = pipeline.source.num_frames
print(f"總影格數: {num_frames}")

results = []

for frame in range(num_frames):
    # 呼叫 OVITO 計算該幀的資料
    data = pipeline.compute(frame)
    timestep = data.attributes['Timestep']
    time_ps = timestep * DT_PS
    # 提取原子的位移向量陣列 (N x 3 的矩陣)
    # Displacement = [dx, dy, dz]
    disp = data.particles['Displacement']
    
    # 計算每個原子的位移平方: dx^2 + dy^2 + dz^2
    sq_disp = np.sum(disp ** 2, axis=1)
    
    # 計算「均」方位移 (Mean Squared Displacement)
    msd = np.mean(sq_disp)
    
    
    results.append([time_ps, msd])
    
    if frame % 10 == 0 or frame == num_frames - 1:
        print(f"Frame {frame:4d} | Time: {time_ps:8.2f} ps | MSD: {msd:.4f} Å²")

# =========================================================================
# 4. 儲存結果
# =========================================================================
# 轉成 numpy array 並存成 txt，可用於後續作圖或剛剛寫好的 log-binning
results = np.array(results)
np.savetxt(OUTPUT_FILE, results, header="Time(ps) MSD(Angstrom^2)", fmt="%.6f")

print(f"\n✅ 分析完成！MSD 數據已儲存至 {OUTPUT_FILE}")