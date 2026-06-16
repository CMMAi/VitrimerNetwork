import numpy as np
from ovito.io import import_file

# =====================================================================
# 1. 參數設定
# =====================================================================
DATA_FILE = "pre_eq1.data"    # 請確認填入你的 .data 檔名
OUTPUT_FILE = "crosslinks_id.txt"

print(f"正在載入 {DATA_FILE} 並分析網路拓樸連線數...")

# =====================================================================
# 2. 載入檔案與獲取資料
# =====================================================================
pipeline = import_file(DATA_FILE)
data = pipeline.compute()

if data.particles.bonds is None or data.particles.bonds.topology is None:
    raise ValueError("錯誤：找不到化學鍵資訊！請確認這是一個包含 Bonds 的 .data 檔。")

topology_array = np.array(data.particles.bonds.topology)
num_particles = data.particles.count

# =====================================================================
# 3. 拓樸分析：計算每顆原子的連接數 (Degree)
# =====================================================================
degrees = np.bincount(topology_array.flatten(), minlength=num_particles)

# =====================================================================
# 4. 篩選交聯點 Index，並轉換為 Identifier
# =====================================================================
# 第一步：先找出符合條件的 0-based Index (座位號碼)
crosslink_indices = np.where(degrees > 2)[0]

# 第二步：獲取系統中所有原子的 Identifier 陣列 (身分證字號)
identifiers_array = np.array(data.particles['Particle Identifier'])

# 第三步：用 Index 當作鑰匙，抽出對應的 Identifier
crosslink_identifiers = identifiers_array[crosslink_indices]

num_crosslinks = len(crosslink_identifiers)
print(f"拓樸掃描完成！系統中共有 {num_particles} 顆原子。")

if num_crosslinks == 0:
    print("警告：系統中找不到任何連線數大於 2 的原子，請檢查你的網路是否已成功交聯！")
else:
    print(f"找到 {num_crosslinks} 個拓樸交聯點。")
    
    # =====================================================================
    # 5. 匯出 Identifier 的文字檔
    # =====================================================================
    # 這裡匯出的就是 LAMMPS 裡面原始的 Atom ID 了！
    np.savetxt(OUTPUT_FILE, crosslink_identifiers, fmt="%d")
    print(f"檔案已成功儲存為：{OUTPUT_FILE} (Identifier 版本)")