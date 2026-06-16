import numpy as np
from collections import defaultdict, deque

def modify(frame, data):
    # =========================================================
    # 🎯 核心參數設定
    # =========================================================
    ID_FILE = "crosslinks_id.txt" # 🚨 交聯點 Identifier 名單
    
    CUTOFF_LENGTH = 9            # 區分長鏈/短鏈的鍵數閥值
    # =========================================================

    if 'Displacement' not in data.particles:
        raise RuntimeError("🚨 錯誤：請先加入 'Calculate displacements' 修改器！")

    # 1. 初始幀：拓樸掃描與「相對配對」萃取
    if not hasattr(modify, 'initialized'):
        print("🚀 Frame 0: 啟動拓樸掃描，萃取 交聯點-中點 配對 (Node-Midpoint Pairs)...")
        
        target_ids = np.loadtxt(ID_FILE, dtype=int)
        current_ids = data.particles['Particle Identifier']
        bond_topo = data.particles.bonds.topology
        
        target_ids_set = set(target_ids)
        id_to_idx = {pid: idx for idx, pid in enumerate(current_ids)}
        crosslink_indices = {id_to_idx[pid] for pid in target_ids_set if pid in id_to_idx}
        
        adj = defaultdict(list)
        for u, v in bond_topo:
            adj[u].append(v)
            adj[v].append(u)
            
        short_pairs = []
        long_pairs  = []
        
        # 邊追蹤演算法尋找有效網鏈
        for start_node in crosslink_indices:
            queue = deque([(start_node, [start_node])])
            visited = {start_node}
            
            while queue:
                curr, path = queue.popleft()

                    
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        if neighbor in crosslink_indices:
                            # 為了不重複計算，只看 ID 從小到大的路徑
                            if start_node < neighbor:
                                full_path = path + [neighbor]
                                strand_len = len(full_path) - 1
                                
                                if strand_len >= 2:
                                    mid_index = full_path[len(full_path) // 2]
                                    
                                    node_A_id = current_ids[full_path[0]]
                                    node_B_id = current_ids[full_path[-1]]
                                    mid_id = current_ids[mid_index]
                                    
                                    # 🌟 為了增加統計量，一條網鏈我們提取兩組配對：
                                    # (交聯點A, 中點) 與 (交聯點B, 中點)
                                    if strand_len <= CUTOFF_LENGTH:
                                        short_pairs.append((node_A_id, mid_id))
                                        short_pairs.append((node_B_id, mid_id))
                                    else:
                                        long_pairs.append((node_A_id, mid_id))
                                        long_pairs.append((node_B_id, mid_id))
                        else:
                            queue.append((neighbor, path + [neighbor]))

        # 將配對矩陣轉為 NumPy Array 供後續高速運算
        modify.short_pairs = np.array(short_pairs, dtype=int) if short_pairs else np.empty((0,2), dtype=int)
        modify.long_pairs = np.array(long_pairs, dtype=int) if long_pairs else np.empty((0,2), dtype=int)
        modify.initialized = True
        
        print(f"✅ 分類完成！")
        print(f"   - 短鏈 配對數量: {len(modify.short_pairs)}")
        print(f"   - 長鏈 配對數量: {len(modify.long_pairs)}")

    # =========================================================
    # 2. 計算當前幀 (Frame t) 的 Dynamic Internal MSD
    # =========================================================
    disp = data.particles['Displacement']
    current_ids = data.particles['Particle Identifier']
    
    # 建立 ID -> Index 的極速翻譯字典
    id_to_idx = {pid: idx for idx, pid in enumerate(current_ids)}
    translate_func = np.vectorize(lambda x: id_to_idx.get(x, -1))
    
    def calc_internal_msd(pairs_array):
        if len(pairs_array) == 0: return 0.0
        
        # 找出配對原子目前的 Index
        node_indices = translate_func(pairs_array[:, 0])
        mid_indices = translate_func(pairs_array[:, 1])
        
        # 過濾掉因為某些原因在當前畫面遺失的原子
        valid_mask = (node_indices != -1) & (mid_indices != -1)
        valid_nodes = node_indices[valid_mask]
        valid_mids = mid_indices[valid_mask]
        
        if len(valid_nodes) == 0: return 0.0
        
        # 提取位移向量 d(t)
        d_nodes = disp[valid_nodes]
        d_mids = disp[valid_mids]
        
        # 🌟 核心物理運算：相對位移相減
        d_rel = d_mids - d_nodes
        
        # 算出平方和並取平均
        return np.mean(np.sum(d_rel**2, axis=1))

    # 計算結果
    int_msd_short = calc_internal_msd(modify.short_pairs)
    int_msd_long = calc_internal_msd(modify.long_pairs)
    
    # 寫入屬性，供 Data Table 匯出
    data.attributes['Int_MSD_Short'] = int_msd_short
    data.attributes['Int_MSD_Long'] = int_msd_long
    
    if frame % 10 == 0:
        print(f"Frame {frame}: Int_MSD_Short = {int_msd_short:.3f} | Int_MSD_Long = {int_msd_long:.3f}")