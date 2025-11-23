import pandas as pd
import numpy as np

# 1. 读取原始 excel（把路径改成你自己的文件）
input_file = "te.xlsx"      # 原文件名
output_file = "output3.xlsx"    # 输出文件名

df = pd.read_excel(input_file, sheet_name='24')

# 假设原表中列名为 pred_cx、pred_cy
cx_sim = df["CIE_x"]
cy_sim = df["CIE_y"]

# 2. 计算 cx_true（图一）
#    cx_true = 0.3646494751042694 - 0.00189118604297041 * cx_sim
df["cx_true"] = 0.3646494751042694 - 0.00189118604297041 * cx_sim

# 3. 计算 cy_true（图二）
# cy_true = (
#   0.124210965653585 * cy_sim
# + 0.0223147560703659 * exp(0.0038062335248655 * cy_sim)
# + 0.0387995784040606 * exp(0.0124224941828288 * cy_sim)
# + 0.36094897158125
# - 0.00613138512159672 * exp(-0.203210615896626 * cy_sim)
# )
df["cy_true"] = (
    0.124210965653585 * cy_sim
    + 0.0223147560703659 * np.exp(0.0038062335248655 * cy_sim)
    + 0.0387995784040606 * np.exp(0.0124224941828288 * cy_sim)
    + 0.36094897158125
    - 0.00613138512159672 * np.exp(-0.203210615896626 * cy_sim)
)

# 4. 保存为新的 xlsx
df.to_excel(output_file, index=False)

print("计算完成，结果已保存到：", output_file)
