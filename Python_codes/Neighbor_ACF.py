import numpy as np
from collections import defaultdict, deque

def modify(frame, data):
    # =========================================================
    # 🎯 核心設定區
    # =========================================================
    ID_FILE = "crosslinks_id.txt"  # 你匯出的 LAMMPS Identifier 檔案
    # =========================================================

    # 極速版內部函數：客製化 BFS
    def get_neighbor_pairs_fast(current_ids, target_ids, topo):
        # 1. 建立 ID 映射 (Set 尋找速度 O(1))
        target_ids_set = set(target_ids)
        id_to_idx = {pid: idx for idx, pid in enumerate(current_ids)}
        
        crosslink_indices = set()
        for pid in target_ids_set:
            if pid in id_to_idx:
                crosslink_indices.add(id_to_idx[pid])
                
        # 2. 建立原生相鄰矩陣 (Adjacency List) -> 捨棄笨重的 NetworkX
        adj = defaultdict(list)
        for u, v in topo:
            adj[u].append(v)
            adj[v].append(u)
            
        pairs = set()
        
        # 3. 針對每一個交聯點進行客製化 BFS
        for start_node in crosslink_indices:
            # Queue 內儲存: (當前節點, 當前深度)
            queue = deque([(start_node, 0)])
            visited = {start_node}
            
            while queue:
                curr, depth = queue.popleft()
                
                    
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        
                        if neighbor in crosslink_indices:
                            # 物理核心加速：我們撞到了另一個交聯點！
                            # 記錄這對鄰居，且「絕對不要」把這個 neighbor 放進 queue 裡！
                            # 這完美保證了路徑不會穿透交聯點，並瞬間砍掉 99% 的無效搜尋空間
                            u_id = current_ids[start_node]
                            v_id = current_ids[neighbor]
                            pairs.add((min(u_id, v_id), max(u_id, v_id)))
                        else:
                            # 這是普通的網鏈原子，繼續往下探索
                            queue.append((neighbor, depth + 1))
        return pairs

    # 讀取當前幀資料
    current_identifiers = data.particles['Particle Identifier']
    bond_topo = data.particles.bonds.topology

    # 1. 初始幀：建立「初始鄰居名單」
    if not hasattr(modify, 'initial_pairs'):
        print(f" Frame 0: 正在建構初始拓樸，提取初始鄰居對 (極速版)...")
        target_identifiers = np.loadtxt(ID_FILE, dtype=int)
        
        # 取得 t=0 時的所有鄰居對
        initial_pairs = get_neighbor_pairs_fast(current_identifiers, target_identifiers, bond_topo)
        modify.initial_pairs = initial_pairs
        modify.target_ids = target_identifiers
        
        print(f" 成功記錄了 {len(initial_pairs)} 對初始交聯點鄰居。")

    initial_pairs = modify.initial_pairs
    num_initial = len(initial_pairs)
    
    if num_initial == 0:
        data.attributes['C_neighbor'] = 0.0
        return

    # 2. 計算當前幀的 C_neighbor(t)
    if frame == 0:
        c_val = 1.0
        surviving_pairs = initial_pairs
    else:
        # 重新掃描當前幀的拓樸
        current_pairs = get_neighbor_pairs_fast(current_identifiers, modify.target_ids, bond_topo)
        
        # 計算交集
        surviving_pairs = initial_pairs.intersection(current_pairs)
        c_val = len(surviving_pairs) / num_initial
        
    # 3. 寫入屬性
    data.attributes['C_neighbor'] = c_val
    
    print(f"   Frame {frame}: C_neighbor(t) = {c_val:.5f} (殘存對數: {len(surviving_pairs)})")