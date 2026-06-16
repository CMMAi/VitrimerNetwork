import sys

def write_ovito_dump(file_handle, timestep, atom_ids):
    """將原子 ID 寫成標準的 LAMMPS dump 格式，賦予屬性 c_reacting = 1"""
    file_handle.write(f"ITEM: TIMESTEP\n{timestep}\n")
    file_handle.write(f"ITEM: NUMBER OF ATOMS\n{len(atom_ids)}\n")
    # 寫入一個虛擬的 Box Bounds (OVITO 讀取 subset 時需要這個格式，但不影響主軌跡)
    file_handle.write("ITEM: BOX BOUNDS pp pp pp\n0.0 1.0\n0.0 1.0\n0.0 1.0\n")
    # 輸出 id 以及一個自訂義的選擇標籤 c_reacting
    file_handle.write("ITEM: ATOMS id c_reacting\n")
    for atom_id in sorted(atom_ids):
        file_handle.write(f"{atom_id} 1\n")

def analyze_and_export_ovito(dump_filename, transient_out, cumulative_out):
    print(f"開始解析軌跡檔: {dump_filename}")
    
    current_timestep = -1
    previous_bonds = set()
    cumulative_atoms = set()
    is_first_frame = True
    
    frame_count = 0
    
    with open(dump_filename, 'r') as f_in, \
         open(transient_out, 'w') as f_trans, \
         open(cumulative_out, 'w') as f_cumul:
        
        while True:
            line = f_in.readline()
            if not line:
                break # 檔案結束
                
            if line.startswith("ITEM: TIMESTEP"):
                current_timestep = int(f_in.readline().strip())
                frame_count += 1
                
            elif line.startswith("ITEM: NUMBER OF ENTRIES"):
                num_entries = int(f_in.readline().strip())
                
            elif line.startswith("ITEM: ENTRIES"):
                current_bonds = set()
                
                # 讀取這一幀的化學鍵
                for _ in range(num_entries):
                    data = f_in.readline().split()
                    # 假設 index atom1 atom2 type (請依你的 dump local 格式確認 index)
                    atom1 = int(data[1])
                    atom2 = int(data[2])
                    bond = tuple(sorted((atom1, atom2)))
                    current_bonds.add(bond)
                
                transient_atoms = set()
                
                # 第一幀沒有「上一幀」可以比對，所以跳過計算差異
                if is_first_frame:
                    is_first_frame = False
                else:
                    broken_bonds = previous_bonds - current_bonds
                    formed_bonds = current_bonds - previous_bonds
                    
                    # 收集當下正在斷裂或生成的原子
                    for b in broken_bonds | formed_bonds:
                        transient_atoms.update(b)
                
                # 更新累積反應的原子名單
                cumulative_atoms.update(transient_atoms)
                
                # 分別寫入兩個 OVITO dump 檔
                write_ovito_dump(f_trans, current_timestep, transient_atoms)
                write_ovito_dump(f_cumul, current_timestep, cumulative_atoms)
                
                previous_bonds = current_bonds

    print(f"解析完成！總共處理了 {frame_count} 幀。")
    print(f"當下反應原子軌跡已存至: {transient_out}")
    print(f"累積反應原子軌跡已存至: {cumulative_out}")

# ==========================================
# 執行區域
# ==========================================
if __name__ == "__main__":
    input_file = "bonds.dump"   
    transient_file = "ovito_transient.dump"
    cumulative_file = "ovito_cumulative.dump"
    
    analyze_and_export_ovito(input_file, transient_file, cumulative_file)