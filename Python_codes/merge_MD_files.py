import glob
import gzip
import re

def natural_sort_key(s):
    """
    實作自然排序，確保檔名中的數字能正確排列 (例如: run_2 會排在 run_10 前面)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def merge_lammps_gz(search_pattern, output_filename):
    """
    搜尋符合 pattern 的 LAMMPS gz 軌跡檔，進行自然排序並無縫去重串聯。
    """
    # 1. 使用 glob 尋找所有符合條件的檔案
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"找不到符合條件的檔案: {search_pattern}")
        return

    # 2. 使用自然排序對檔案進行排列
    files = sorted(files, key=natural_sort_key)
    
    print(f"找到 {len(files)} 個檔案，開始串聯序列...")
    for f in files:
        print(f"   - {f}")
        
    seen_timesteps = set()
    
    # 3. 開啟輸出的 gz 檔案 (寫入模式，壓縮層級 6 兼顧速度與大小)
    with gzip.open(output_filename, 'wt', compresslevel=6) as fout:
        for filepath in files:
            print(f"正在處理: {filepath} ...", end="", flush=True)
            
            added_frames = 0
            skipped_frames = 0
            
            # 開啟輸入的 gz 檔案 (讀取模式)
            with gzip.open(filepath, 'rt') as fin:
                write_current_frame = False
                
                for line in fin:
                    if line.startswith("ITEM: TIMESTEP"):
                        # 讀取下一行來獲取具體的 timestep 數字
                        timestep_line = next(fin)
                        current_ts = int(timestep_line.strip())
                        
                        # 檢查這個 timestep 是否已經處理過
                        if current_ts not in seen_timesteps:
                            seen_timesteps.add(current_ts)
                            write_current_frame = True
                            added_frames += 1
                            
                            # 寫入 ITEM: TIMESTEP 和數字行
                            fout.write(line)
                            fout.write(timestep_line)
                        else:
                            # 發現重複的 timestep，關閉寫入開關，直到遇到下一個 timestep
                            write_current_frame = False
                            skipped_frames += 1
                            
                    elif write_current_frame:
                        # 如果開關是開啟的，把該 frame 的其他資料行寫入
                        fout.write(line)
                        
            print(f" 完成！(新增 {added_frames} 幀, 捨棄 {skipped_frames} 個重複幀)")

    print("-" * 50)
    print(f"完美串聯！總共寫入了 {len(seen_timesteps)} 個獨立的 frames。")
    print(f"檔案已儲存為: {output_filename}")


if __name__ == "__main__":
    # =====================================================================
    # 請在這裡修改你的搜尋條件與輸出檔名
    # =====================================================================
    
    for dir in ["xy", "yz", "xz"]:
        print(f"正在處理資料夾: {dir} ...")
        # 你想尋找的檔案特徵 (支援 * 萬用字元)
        # 例如: "basename_*.dump.gz" 或 "production_run_*.bond.gz"
        TARGET_TRAJ_WO = f"SRT3_{dir}/trajectory_450_SRT_wo_log_*.lammpstrj.gz"
        TARGET_TRAJ_RX = f"SRT3_{dir}/trajectory_450_SRT_rx_log_*.lammpstrj.gz"

        TARGET_BOND_WO = f"SRT3_{dir}/bonds_450_SRT_wo_1_log_*.local.gz"
        TARGET_BOND_RX = f"SRT3_{dir}/bonds_450_SRT_rx_1_log_*.local.gz"
        # 合併後輸出的完整檔名
        OUTPUT_TRAJ_WO = f"SRT3_{dir}/Traj_wo_merged.{dir}.lammpstrj.gz"
        OUTPUT_TRAJ_RX = f"SRT3_{dir}/Traj_rx_merged.{dir}.lammpstrj.gz"

        OUTPUT_BOND_WO = f"SRT3_{dir}/Bonds_wo_merged.{dir}.local.gz"
        OUTPUT_BOND_RX = f"SRT3_{dir}/Bonds_rx_merged.{dir}.local.gz"

        # =====================================================================
        
        # 執行合併
        merge_lammps_gz(TARGET_TRAJ_WO, OUTPUT_TRAJ_WO)
        merge_lammps_gz(TARGET_TRAJ_RX, OUTPUT_TRAJ_RX)
        merge_lammps_gz(TARGET_BOND_WO, OUTPUT_BOND_WO)
        merge_lammps_gz(TARGET_BOND_RX, OUTPUT_BOND_RX)