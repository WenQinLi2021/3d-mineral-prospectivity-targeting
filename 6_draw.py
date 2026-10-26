# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

# # 读取预测
# pred = pd.read_csv("xy_proba.tsv", sep="\t")

# # 全局颜色归一化
# pmin, pmax = float(pred["proba"].min()), float(pred["proba"].max())
# norm = Normalize(vmin=pmin, vmax=pmax)
# X=pred['ix']
# Y=pred['iy']
# Z=pred['iz']

# # 子集阈值与下采样
# thr = 0.7
# mask = (pred["proba"].to_numpy()) >= thr
# idx = np.where(mask)[0]
# if idx.size > 100_000:
#     idx = np.random.default_rng(0).choice(idx, size=100_000, replace=False)

# # 绘制网格坐标的 3D 散点；公开版本不恢复绝对坐标
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(X[idx], Y[idx], Z[idx],
#                 c=pred["proba"].to_numpy()[idx],
#                 s=1, alpha=0.5, norm=norm, cmap="viridis")

# ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
# ax.set_title(f"High-probability voxels (proba≥{thr})")
# cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
# cbar.set_label("Prospectivity (proba)")

# # 可选：调整视角
# ax.view_init(elev=90, azim=-90)
# plt.show()






# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
# import sys
# sys.stdout
# # ================= 1. 读取预测结果 =================
# pred = pd.read_csv("historical_baseline.tsv", sep="\t")

# X = pred["X"].to_numpy()
# Y = pred["Y"].to_numpy()
# Z = pred["Z"].to_numpy()
# P = pred["proba"].to_numpy()

# print("原始概率分布：")
# for q in [0.8, 0.85, 0.9, 0.95, 0.99]:
#     print(f">= {q} : {(P >= q).mean()*100:.2f}%")

# # ================= 2. 分位数重标定（方案一核心） =================

# # 你希望的“合理异常比例层级”
# q80 = np.quantile(P, 0.80)
# q90 = np.quantile(P, 0.90)
# q95 = np.quantile(P, 0.95)
# q99 = np.quantile(P, 0.99)

# P_new = np.zeros_like(P)

# # 低值区：整体压缩
# m = P < q80
# P_new[m] = 0.4 * (P[m] / q80)

# # 次级异常
# m = (P >= q80) & (P < q90)
# P_new[m] = 0.4 + 0.2 * (P[m] - q80) / (q90 - q80)

# # 强异常
# m = (P >= q90) & (P < q95)
# P_new[m] = 0.6 + 0.15 * (P[m] - q90) / (q95 - q90)

# # 很强异常
# m = (P >= q95) & (P < q99)
# P_new[m] = 0.75 + 0.15 * (P[m] - q95) / (q99 - q95)

# # 核心靶区（Top 1%）
# m = (P >= q99)
# P_new[m] = 0.90 + 0.10 * (P[m] - q99) / (P.max() - q99 + 1e-6)

# # 裁剪到 [0,1]
# P_new = np.clip(P_new, 0, 1)

# pred["proba_calib"] = P_new

# print("\n标定后概率分布：")
# for q in [0.6, 0.7, 0.8, 0.9, 0.95]:
#     print(f">= {q} : {(P_new >= q).mean()*100:.2f}%")

# # ================= 3. 选择高异常区用于绘图 =================

# thr = 0.7   # 现在用标定后的阈值
# mask = P_new >= thr
# idx = np.where(mask)[0]

# # 下采样，防止点太多
# if idx.size > 100_000:
#     idx = np.random.default_rng(0).choice(idx, size=100_000, replace=False)

# # 颜色归一化（基于标定后的值）
# norm = Normalize(vmin=P_new[idx].min(), vmax=P_new[idx].max())

# # ================= 4. 3D 可视化（加速 + 工程友好） =================

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection="3d")

# sc = ax.scatter(
#     X[idx], Y[idx], Z[idx],
#     c=P_new[idx],          # 🔥 用“标定后的概率”
#     s=1,
#     alpha=0.6,
#     cmap="viridis",
#     norm=norm,
#     marker=".",
#     linewidths=0,
#     rasterized=True       # 关键加速
# )

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title(f"Calibrated prospectivity (Top {(P_new>=thr).mean()*100:.2f}%)")

# cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
# cbar.set_label("Calibrated prospectivity")

# # 俯视更像平面异常图
# ax.view_init(elev=90, azim=-90)

# plt.show()




# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize

# pred = pd.read_csv("historical_baseline.tsv", sep="\t")

# # 一次性转 numpy（避免重复 to_numpy）
# X = pred["X"].to_numpy()
# Y = pred["Y"].to_numpy()
# Z = pred["Z"].to_numpy()
# P = pred["proba"].to_numpy()

# # 归一化
# norm = Normalize(vmin=float(P.min()), vmax=float(P.max()))

# # 阈值 + 下采样
# thr = 0.7
# idx = np.where(P >= thr)[0]

# if idx.size > 100_000:
#     idx = np.random.default_rng(0).choice(idx, size=100_000, replace=False)

# # ===== 绘图（关键优化点）=====
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection="3d")

# sc = ax.scatter(
#     X[idx], Y[idx], Z[idx],
#     c=P[idx],
#     s=1,
#     alpha=0.5,
#     cmap="viridis",
#     norm=norm,
#     marker=".",          # 比默认快
#     linewidths=0,
#     rasterized=True      # 🔥 关键：直接栅格化，速度提升 2–5 倍
# )

# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_title(f"High-probability voxels (proba≥{thr})")

# cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
# cbar.set_label("Prospectivity")

# ax.view_init(elev=90, azim=-90)
# plt.show()



















# import pandas as pd

# attr = pd.read_csv("属性_with_label.tsv", sep="\t", encoding="utf-8-sig")
# pred = pd.read_csv("pu_predictions.tsv", sep="\t", encoding="utf-8-sig")
# x0, y0, z0 = ...  # restricted absolute coordinates are supplied separately
# DX, DY, DZ_eff = 60, 60, 0.5

# # 直接覆盖 ix/iy/iz 三列
# pred["ix"] = x0 + (pred["ix"] - 59) * DX
# pred["iy"] = y0 + (pred["iy"] - 39) * DY
# pred["iz"] = z0 - pred["iz"] * DZ_eff  # 顶面向下递增

# # 如果希望改列名为 X/Y/Z（推荐这样更直观）
# pred = pred.rename(columns={"ix": "X", "iy": "Y", "iz": "Z"})

# pred["X"] = attr["X"] 
# pred["Y"] = attr["Y"] 
# pred["Z"] = attr["Z"]    # 直接替换

# pred.to_csv("pu_predictions.tsv", sep="\t", index=False, encoding="utf-8-sig")


# import pandas as pd
# import pyarrow.parquet as pq
# import pyarrow as pa
# import sys

# sys.stdout.reconfigure(encoding="utf-8")

# ATTR_PARQUET = "属性.parquet"
# PRED_PARQUET = "pu_predictions.parquet"
# OUT_PARQUET  = "pu_with_xyz.parquet"

# CHUNK = 1_000_000

# ATTR_COLS = ["index", "X", "Y", "Z", "地层"]

# print("📦 预加载属性映射 ...")
# pf_attr = pq.ParquetFile(ATTR_PARQUET)

# attr_map = {}
# for batch in pf_attr.iter_batches(batch_size=CHUNK, columns=ATTR_COLS):
#     df = batch.to_pandas()
#     for row in df.itertuples(index=False):
#         attr_map[row.index] = (row.X, row.Y, row.Z, row.地层)

# print(f"✅ 属性映射加载完成：{len(attr_map):,} 行")

# print("🧾 开始 Parquet → Parquet 合并 ...")
# pf_pred = pq.ParquetFile(PRED_PARQUET)

# writer = None

# for i, batch in enumerate(pf_pred.iter_batches(batch_size=CHUNK)):
#     df = batch.to_pandas()

#     xs, ys, zs, layers = [], [], [], []
#     for idx in df["index"]:
#         x, y, z, layer = attr_map.get(idx, (None, None, None, None))
#         xs.append(x)
#         ys.append(y)
#         zs.append(z)
#         layers.append(layer)

#     df.insert(1, "X", xs)
#     df.insert(2, "Y", ys)
#     df.insert(3, "Z", zs)
#     df.insert(4, "地层", layers)

#     table = pa.Table.from_pandas(df, preserve_index=False)

#     if writer is None:
#         writer = pq.ParquetWriter(
#             OUT_PARQUET,
#             table.schema,
#             compression="zstd"
#         )

#     writer.write_table(table)
#     print(f"✅ batch {i+1} 完成")

# if writer:
#     writer.close()

# print("🎉 完成输出：", OUT_PARQUET)



import pandas as pd

#=== 1. 读取 A 和 B ===
A = pd.read_csv("属性_fixed.txt", sep='\t', encoding='utf-8')
B = pd.read_csv("xy_proba.tsv", sep='\t', encoding='utf-8')

#=== 2. 从 A 取 index 与 XYZ ===
cols_to_add = ["index", "x", "y", "z","地层"]
A_sub = A[cols_to_add]

#=== 3. 删掉 B 的 ix iy iz ===
cols_to_drop = ["ix", "iy", "iz"]
for c in cols_to_drop:
    if c in B.columns:
        B = B.drop(columns=c)

#=== 4. 合并：按行对应（index 相同长度，行索引对应） ===
B_merged = pd.concat([A_sub, B], axis=1)

#=== 5. 输出结果 ===
B_merged.to_csv("historical_baseline.tsv", sep='\t', index=False, encoding='utf-8')
print("Done!")
