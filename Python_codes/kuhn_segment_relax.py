import numpy as np
import networkx as nx

# 為了在多執行緒中安全快取，我們將變數綁定在函式本身 (Function attribute)
def modify(frame, data):
    STEP_SIZE = 1  # Kuhn Segment 跨越的鍵數
    
    # 防禦機制：確保使用者有加 Calculate Displacements
    if 'Displacement' not in data.particles:
        raise RuntimeError("錯誤：請在 Python Script Modifier 之前，先加入 'Calculate displacements' Modifier！")
    
    # =====================================================================
    # 1. 拓樸快取 (只在第一次呼叫時執行，避免每幀重複計算)
    # =====================================================================
    if not hasattr(modify, 'segment_heads'):
        print(f"首次執行：建構圖論模型並擷取 Kuhn Segments (Step={STEP_SIZE})...")
        bond_topo = data.particles.bonds.topology
        G = nx.Graph()
        G.add_edges_from(bond_topo)
        
        heads, tails = [], []
        # DFS 深度優先搜尋
        for start_node in G.nodes():
            stack = [(start_node, [start_node])]
            while stack:
                curr, path = stack.pop()
                if len(path) == STEP_SIZE + 1:
                    if path[0] < path[-1]: # 避免反向重複
                        heads.append(path[0])
                        tails.append(path[-1]) # 精準抓取尾端
                    continue
                for neighbor in G.neighbors(curr):
                    if neighbor not in path:
                        if G.degree(neighbor) > 2 and len(path) < STEP_SIZE:
                            continue
                        stack.append((neighbor, path + [neighbor]))
                        
        # 快取到函式物件上
        modify.segment_heads = np.array(heads)
        modify.segment_tails = np.array(tails)
        print(f"成功擷取 {len(heads)} 個線段！")

    heads = modify.segment_heads
    tails = modify.segment_tails

    # =====================================================================
    # 2. 穿越時空：利用位移反推 t=0 的精準座標
    # =====================================================================
    pos_t = data.particles['Position']
    disp_t = data.particles['Displacement']
    
    # 神奇的還原術：p(0) = p(t) - displacement(t)
    pos_0 = pos_t - disp_t 
    
    # 算出現狀 u(t) 與 基準 u(0) 的向量
    vec_t = pos_t[tails] - pos_t[heads]
    vec_0 = pos_0[tails] - pos_0[heads]
    
    # 單位化
    len_t = np.linalg.norm(vec_t, axis=1, keepdims=True)
    len_0 = np.linalg.norm(vec_0, axis=1, keepdims=True)
    len_t[len_t == 0] = 1.0
    len_0[len_0 == 0] = 1.0
    
    u_t = vec_t / len_t
    u_0 = vec_0 / len_0
    
    # =====================================================================
    # 3. 計算 P2(t)
    # =====================================================================
    if frame == 0:
        p2_val = 1.0
    else:
        dot_prod = np.sum(u_t * u_0, axis=1)
        p2_val = np.mean(0.5 * (3.0 * dot_prod**2 - 1.0))
        
    # 將數據寫入屬性中
    data.attributes['P2_Autocorr'] = p2_val
    
    if frame % 10 == 0:
        print(f"   Frame {frame}: P2 = {p2_val:.5f}")