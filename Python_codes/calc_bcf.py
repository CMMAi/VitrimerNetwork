#!/usr/bin/env python3
"""
calc_bcf_streamlined.py — 極簡版拓樸演化分析
特性：針對已預先合併的單一軌跡檔，自動將第 0 幀作為拓樸比較基準 (t=0)。
"""

import os, re, csv, gzip
from collections import deque
from typing import List, Tuple, Set, Optional

TIMESTEP_RE = re.compile(r'(-?\d+)')

# ---------------------------------------------------------
# Robust I/O Classes & Functions (保留核心讀取能力)
# ---------------------------------------------------------
class LineStream:
    def __init__(self, iterable):
        self.it = iter(iterable)
        self.buf = deque()

    def next(self) -> Optional[str]:
        if self.buf:
            return self.buf.popleft()
        return next(self.it, None)

    def push(self, line: Optional[str]):
        if line is not None:
            self.buf.appendleft(line)

def parse_timestep(ls: LineStream, first_line: str) -> Optional[int]:
    rest = first_line.replace("ITEM: TIMESTEP", "", 1)
    m = TIMESTEP_RE.search(rest)
    if m:
        try: return int(m.group(1))
        except: pass
    while True:
        ln = ls.next()
        if ln is None: return None
        if ln.strip() == "": continue
        m = TIMESTEP_RE.search(ln)
        if m:
            try: return int(m.group(1))
            except: return None
        if ln.startswith("ITEM: "):
            ls.push(ln)
            return None

def parse_entries_block(ls: LineStream, include_type: bool, filter_type: Optional[int], 
                        col_type: int, col_i: int, col_j: int) -> Set[tuple]:
    bonds: Set[tuple] = set()
    max_col_needed = max(col_type, col_i, col_j)
    crosslink = []
    with open("crosslinks_id.txt", "r") as f:
        for line in f:
            crosslink.append(int(line))
    crosslink_set = set(crosslink)

    while True:
        l = ls.next()
        if l is None:
            break
        if l.startswith("ITEM: "):
            ls.push(l)
            break
        parts = l.strip().split()
        if len(parts) <= max_col_needed:
            continue
        try:
            btype = int(parts[col_type])
            i = int(parts[col_i])
            j = int(parts[col_j])
        except ValueError:
            continue
            
        if (filter_type is not None) and (btype != filter_type):
            continue
        if (i in crosslink_set) or (j in crosslink_set):
            continue

        # 確保 (i, j) 順序一致，避免方向性干擾
        a, b = (i, j) if i < j else (j, i)
        if include_type:
            bonds.add((btype, a, b))
        else:
            bonds.add((a, b))
    return bonds

def parse_local_dump_gz(path: str, include_type: bool, filter_type: Optional[int], 
                        col_type: int, col_i: int, col_j: int) -> List[Tuple[int, Set[tuple]]]:
    frames = []
    print(f"    正在解析軌跡檔: {path} ...")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        ls = LineStream(f)
        while True:
            line = ls.next()
            if line is None:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            ts = parse_timestep(ls, line)
            if ts is None:
                continue

            while True:
                hdr = ls.next()
                if hdr is None:
                    break
                if hdr.startswith("ITEM: NUMBER OF ENTRIES"):
                    _ = ls.next()
                    continue
                if hdr.startswith("ITEM: BOX BOUNDS"):
                    _ = ls.next(); _ = ls.next(); _ = ls.next()
                    continue
                if hdr.startswith("ITEM: ENTRIES"):
                    bonds = parse_entries_block(ls, include_type, filter_type, col_type, col_i, col_j)
                    frames.append((ts, bonds))
                    break
                if hdr.startswith("ITEM: TIMESTEP"):
                    ls.push(hdr)
                    break
    return frames

# ---------------------------------------------------------
# 物理計算：網路拓樸演化演算法
# ---------------------------------------------------------
def compute_topology_evolution(frames: List[Tuple[int, Set[tuple]]], out_csv: str, dt_ps: float):
    if not frames:
        print("   [ERROR] 沒有讀取到任何資料。")
        return

    # 🌟 核心邏輯：直接拿第 0 個 Frame 作為基準 (t=0)
    t0, init_bonds = frames[0]
    N_initial = len(init_bonds)
    
    if N_initial == 0:
        print("   [ERROR] 第 0 幀沒有任何鍵結！請檢查 filter-type 或欄位設定。")
        return

    print(f"    基準建立完成：Timestep = {t0}, 初始鍵結數 = {N_initial}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["frame_index", "timestep", "time_ps", "Survival_Ratio", 
                  "New_Bond_Ratio", "Jaccard_Similarity", "N_Intact", "N_Broken", "N_Formed"]
        w.writerow(header)

        for idx, (ts, current_bonds) in enumerate(frames):
            intact_bonds = init_bonds.intersection(current_bonds)
            broken_bonds = init_bonds.difference(current_bonds)
            formed_bonds = current_bonds.difference(init_bonds)
            all_bonds_union = init_bonds.union(current_bonds)

            N_current = len(current_bonds)
            
            # 計算物理比例
            survival_ratio = len(intact_bonds) / N_initial
            new_bond_ratio = len(formed_bonds) / N_current if N_current > 0 else 0
            jaccard_sim = len(intact_bonds) / len(all_bonds_union) if len(all_bonds_union) > 0 else 0

            row = [
                idx, 
                ts, 
                (ts - t0) * dt_ps,  # 絕對物理時間 (從 0 開始)
                f"{survival_ratio:.6f}",
                f"{new_bond_ratio:.6f}",
                f"{jaccard_sim:.6f}",
                len(intact_bonds),
                len(broken_bonds),
                len(formed_bonds)
            ]
            w.writerow(row)
            
    print(f"   [OK] 分析完成！結果已寫入 {out_csv} (共處理 {len(frames)} 幀)")

# ---------------------------------------------------------
# 批次執行區塊
# ---------------------------------------------------------
if __name__ == "__main__":
    
    # 🚨 迴圈與檔案命名設定
    directions = ["xy", "xz", "yz"]
    
    # 🚨 物理參數與欄位設定
    DT_PS = 0.005          # 每個 timestep 對應的皮秒 (ps)
    TARGET_BOND = 7     # 若只看特定 Bond Type 請填數字，例如 3
    IGNORE_TYPE = False    # 是否忽略 Type，純看拓樸連接
    COL_TYPE = 0           # Bond Type 的欄位 Index (0-based)
    COL_I = 1              # Atom I 的欄位 Index (0-based)
    COL_J = 2              # Atom J 的欄位 Index (0-based)
    
    for d in directions:
        print(f"\n" + "="*50)
        print(f" 開始分析方向: {d}")
        print("="*50)
        
        # 🚨 請根據你「已經合併好」的軌跡檔名稱進行修改
        # 這裡假設你的檔案叫做 merged_bond_x.local.gz, merged_bond_y.local.gz...
        trajectory_file = f"Trajectories/Bonds_rx_merged.{d}.local.gz"
        out_csv = f"BCF/topology_evolution_{d}.csv"
        
        if not os.path.exists(trajectory_file):
            print(f"   [WARN] 找不到檔案 {trajectory_file}，跳過此方向。")
            continue
            
        # 讀取單一合併檔的所有 Frame
        frames = parse_local_dump_gz(
            path=trajectory_file, 
            include_type=(not IGNORE_TYPE), 
            filter_type=TARGET_BOND, 
            col_type=COL_TYPE, col_i=COL_I, col_j=COL_J
        )
        
        # 計算並輸出
        compute_topology_evolution(frames, out_csv, dt_ps=DT_PS)
        
    print("\n 所有方向的分析皆已順利完成！")