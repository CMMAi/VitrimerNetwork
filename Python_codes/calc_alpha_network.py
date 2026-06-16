import numpy as np
import networkx as nx
from ovito.io import import_file

# =====================================================================
# 1. 核心參數設定區
# =====================================================================
# 🚨 警告：你的檔案必須包含「化學鍵資訊」！
# 如果你的 dump 檔沒有 bonds，請改用 import_file("你的.data檔")
# 然後在 GUI 裡把 dump 當作 modifier 疊加上去，再匯出成包含軌跡的 ovito pipeline
TRAJ_FILE = "DA_RX_XY.ovito" 
STEP_SIZE = 2  # Kuhn Segment 跨越的鍵數 (CG 系統建議 1 或 2)

pipeline = import_file(TRAJ_FILE)

# 取得第 0 幀的資料來建立初始拓樸
data0 = pipeline.compute(0)

if 'Topology' not in data0.particles.bonds:
    raise ValueError("錯誤：軌跡中找不到化學鍵 (Bonds) 資訊！請確保你載入了 .data 檔或含有 bonds 的軌跡。")

# 獲取化學鍵列表 (M x 2 陣列)
bond_topo = data0.particles.bonds.topology

print("🕸️ 正在讀取網路拓樸並建構 Graph...")

# =====================================================================
# 2. 圖論尋路演算法：自動擷取 Kuhn Segments
# =====================================================================
# 建立無向圖
G = nx.Graph()
G.add_edges_from(bond_topo)

valid_segments = []

# 用深度優先搜尋 (DFS) 找出所有長度為 STEP_SIZE 的線性路徑
for start_node in G.nodes():
    stack = [(start_node, [start_node])]
    
    while stack:
        curr, path = stack.pop()
        
        # 如果路徑長度達到了我們設定的跨度 (節點數 = 邊數 + 1)
        if len(path) == STEP_SIZE + 1:
            # 為了避免 (A->B->C) 和 (C->B->A) 被重複計算，限制起點 ID 必須小於終點 ID
            if path[0] < path[-1]:
                valid_segments.append((path[0], path[-1]))
            continue
            
        # 繼續往下走
        for neighbor in G.neighbors(curr):
            if neighbor not in path:
                # 🛡️ 物理防禦機制：
                # 如果這顆鄰居原子是「交聯點」(Degree > 2)，而且我們還沒走到終點
                # 代表這條線段被釘死了，不符合純粹的 backbone 鬆弛，我們直接放棄這條路徑！
                if G.degree(neighbor) > 2 and len(path) < STEP_SIZE:
                    continue 
                    
                stack.append((neighbor, path + [neighbor]))

heads = np.array([p[0] for p in valid_segments])
tails = np.array([p[1] for p in valid_segments])
num_segments = len(heads)

print(f"拓樸解析完成！自動擷取了 {num_segments} 個純淨的線性骨架線段。")

# =====================================================================
# 3. 逐幀計算 P2(t)
# =====================================================================
num_frames = pipeline.source.num_frames
t_data = []
p2_data = []
u0_vectors = None

print(f"開始計算 P2(t) 動力學軌跡... 共 {num_frames} 幀")

for frame in range(num_frames):
    data = pipeline.compute(frame)
    current_time = data.attributes.get('Time', frame)
    pos = np.array(data.particles['Position'])
    
    # 算出所有擷取線段的向量
    vec = pos[tails] - pos[heads]
    
    # 單位化
    lengths = np.linalg.norm(vec, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0 
    u_t = vec / lengths
    
    if frame == 0:
        u0_vectors = u_t.copy()
        p2_val = 1.0
    else:
        dot_prod = np.sum(u_t * u0_vectors, axis=1)
        p2_array = 0.5 * (3.0 * dot_prod**2 - 1.0)
        p2_val = np.mean(p2_array)
        
    t_data.append(current_time)
    p2_data.append(p2_val)
    
    if frame % max(1, num_frames // 10) == 0:
        print(f"   進度: {frame}/{num_frames} 幀完成")

# 儲存 P2 原始數據
t_data = np.array(t_data)
p2_data = np.array(p2_data)
np.savetxt("P2_Bond_Autocorr_Network.dat", np.column_stack((t_data, p2_data)), header="Time(ps) P2(t)", fmt="%.6f")