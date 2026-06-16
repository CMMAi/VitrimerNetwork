import numpy as np
from ovito.io import import_file
from ovito.modifiers import CalculateDisplacementsModifier
from ovito.pipeline import FileSource

# =========================================================================
# 1. 使用者設定區
# =========================================================================
# 【新增】包含 Step 0 初始構型的檔案 (Linear dump)

FILE_Prefix = "trajectory_500_SRT_rx"

REF_FILE = f"{FILE_Prefix}.lammpstrj.gz"

# 包含 Step 10 之後的軌跡檔案 (Log dump)
DUMP_FILE = f"{FILE_Prefix}_log.lammpstrj.gz" 

OUTPUT_FILE = f"{FILE_Prefix}_isf.dat"                    
Q_VALUE = 0.15                                    
DT_PS = 0.005                                      

# =========================================================================
# 2. 載入軌跡與設定外部參考
# =========================================================================
print(f"正在載入主要軌跡: {DUMP_FILE} ...")
pipeline = import_file(DUMP_FILE)

# 🚀 【終極正確寫法】建立一個獨立的 FileSource 節點來讀取外部參考檔
print(f"正在將 {REF_FILE} 設為 t=0 的初始參考基準...")
ref_source = FileSource()
ref_source.load(REF_FILE)

# 加入「計算位移」修改器
modifier = CalculateDisplacementsModifier()

# 將讀取好檔案的 FileSource 餵給位移計算器的 reference
modifier.reference = ref_source 

# 強制關閉最小鏡像約定，信任 unwrap 座標
modifier.minimum_image_convention = False  
pipeline.modifiers.append(modifier)

num_frames = pipeline.source.num_frames
print(f"總共找到 {num_frames} 幀 Log 軌跡。開始計算 q = {Q_VALUE} 1/Å 的 ISF...")

# =========================================================================
# 3. 逐幀計算 ISF
# =========================================================================
# 因為 Log dump 是從 step 10 開始，我們在這裡手動補上 t=0 的完美起點！
time_array = [0.0]
isf_array = [1.0]

for frame in range(num_frames):
    data = pipeline.compute(frame)
    
    # 抓取真實 Timestep
    timestep = data.attributes['Timestep']
    time_ps = timestep * DT_PS
    
    # 1. 抓取所有原子的 3D 位移向量 (相對於 REF_FILE 的第 0 幀)
    disp_vec = np.array(data.particles['Displacement'])
    
    # 2. 質心飄移扣除
    com_drift = np.mean(disp_vec, axis=0)
    true_disp = np.linalg.norm(disp_vec - com_drift, axis=1)
    
    # 3. 計算 q * dr
    qd = Q_VALUE * true_disp
    
    # 4. 計算 sinc 函數並取平均
    isf_values = np.sinc(qd / np.pi)
    isf_mean = np.mean(isf_values)
    
    time_array.append(time_ps)
    isf_array.append(isf_mean)
    
    if frame % 10 == 0 or frame == num_frames - 1:
        print(f"已處理 Frame {frame}/{num_frames-1} (Timestep = {timestep}, Time = {time_ps:.2f} ps) -> ISF = {isf_mean:.4f}")

# =========================================================================
# 4. 儲存數據
# =========================================================================
np.savetxt(OUTPUT_FILE, np.column_stack((time_array, isf_array)), 
           header="Time(ps) ISF", fmt="%.6f")
print(f"\n✅ 計算完成！數據已儲存至 {OUTPUT_FILE}")