import numpy as np
from ovito.io import import_file
from ovito.modifiers import CalculateDisplacementsModifier
from collections import defaultdict

# =========================================================================
# 1. 使用者設定區
# =========================================================================
direction = "xz"
# 你的 local bonds 檔案 (t=0 時的鍵結資訊)
BOND_FILE = "bonds_450_SRT_rx_0.local" 
# 你的軌跡檔
DUMP_FILE = f"SRT2_{direction}/trajectory_450_SRT_wo_log.lammpstrj.gz" 
OUTPUT_FILE = f"ISF_450_{direction}_wo_multi_q_largest_network.dat"               
DT_PS = 0.005

R_LIST = np.linspace(20, 75, 5) 
Q_LIST = 2 * np.pi / R_LIST 

# =========================================================================
# 2. 核心技術：從 .local 檔建立 Graph 並尋找最大網路 (Gel)
# =========================================================================
print("🔍 正在解析 local bond 檔案，尋找初始最大網路...")
adj_list = defaultdict(list)

with open(BOND_FILE, "r") as f:
    read_data = False
    for line in f:
        if line.startswith("ITEM: ENTRIES"):
            read_data = True
            continue
        elif line.startswith("ITEM:"):
            read_data = False
            continue
            
        if read_data:
            parts = line.strip().split()
            if len(parts) >= 3:
                # parts[1] 和 parts[2] 是相連的兩個原子 ID
                u, v = int(parts[1]), int(parts[2])
                adj_list[u].append(v)
                adj_list[v].append(u)

# 使用 BFS (廣度優先搜尋) 找出所有獨立的 Cluster
visited = set()
largest_cluster = set()

for node in adj_list.keys():
    if node not in visited:
        queue = [node]
        current_cluster = set([node])
        visited.add(node)
        
        while queue:
            curr = queue.pop(0)
            for neighbor in adj_list[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_cluster.add(neighbor)
                    queue.append(neighbor)
                    
        # 更新最大網路
        if len(current_cluster) > len(largest_cluster):
            largest_cluster = current_cluster

gel_atom_ids = np.array(list(largest_cluster))
print(f"✅ 成功找到最大網路！包含 {len(gel_atom_ids)} 顆原子。")

# =========================================================================
# 3. 載入軌跡並設定高速 Mask
# =========================================================================
print(f"🎬 正在載入軌跡檔案...")
pipeline = import_file(DUMP_FILE)

# 直接計算位移
disp_modifier = CalculateDisplacementsModifier(minimum_image_convention=False)
pipeline.modifiers.append(disp_modifier)

num_frames = pipeline.source.num_frames
print(f"準備計算 {len(Q_LIST)} 個不同的 q 值 (僅針對最大網路)...")

time_array = [0.0]
isf_matrix = [np.ones(len(Q_LIST))] 

# 為了極速過濾原子，建立一個布林對照表 (Lookup table)
max_id = max(gel_atom_ids)
# 假設軌跡裡的原子 ID 可能更大，我們在迴圈內動態對應

for frame in range(num_frames):
    data = pipeline.compute(frame)
    timestep = data.attributes.get('Timestep', frame)
    time_ps = timestep * DT_PS
    
    # 取得當前幀所有原子的 ID 與位移
    p_ids = np.array(data.particles['Particle Identifier'])
    disp_all = np.array(data.particles['Displacement'])
    
    # 🌟 極速篩選：只抓出屬於最大網路的原子
    # np.isin 速度很快，能精準比對 ID
    gel_mask = np.isin(p_ids, gel_atom_ids)
    disp_vec = disp_all[gel_mask]
    
    # 扣除主網路的質心飄移
    com_drift = np.mean(disp_vec, axis=0)
    true_disp = np.linalg.norm(disp_vec - com_drift, axis=1) # shape: (N_gel_atoms,)
    
    # 矩陣魔法：計算 q*d 並取 ISF
    qd_matrix = true_disp[:, np.newaxis] * Q_LIST
    isf_values = np.mean(np.sinc(qd_matrix / np.pi), axis=0)
    
    time_array.append(time_ps)
    isf_matrix.append(isf_values)
    
    if frame % 10 == 0 or frame == num_frames - 1:
        print(f"已處理 Frame {frame} (Time = {time_ps:.2f} ps)")

# =========================================================================
# 4. 儲存數據
# =========================================================================
final_data = np.column_stack((time_array, isf_matrix))
header_str = "Time(ps) " + " ".join([f"q={q:.4f}" for q in Q_LIST])
np.savetxt(OUTPUT_FILE, final_data, header=header_str, fmt="%.6f")
print(f"\n🎉 計算完成！純淨的主網路 ISF 數據已儲存至 {OUTPUT_FILE}")