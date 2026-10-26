import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ================= 配置区 =================
FILE_IN  = "属性_with_label.parquet"
FILE_OUT = "属性_grid.parquet"

COL_X, COL_Y, COL_Z = "x", "y", "z"
DZ_NOMINAL = 0.5

CHUNK = 500_000

# ================= 工具函数 =================
def robust_step(values):
    u = np.sort(pd.unique(values))
    if len(u) < 2:
        return 1.0
    diffs = np.diff(u)
    diffs = diffs[diffs > 0]
    q1, q99 = np.quantile(diffs, [0.01, 0.99])
    diffs = diffs[(diffs >= q1) & (diffs <= q99)]
    return float(np.median(diffs))

def snap_and_index(series, origin, step):
    diffs = series.diff().dropna()
    mad = np.median(np.abs(diffs - np.median(diffs)))
    tol = max(0.45 * step, 3 * mad)

    idx_f = (series - origin) / step
    idx = np.rint(idx_f).astype(np.int64)
    snapped = origin + idx * step

    bad = np.abs(snapped - series) > tol
    return idx, snapped, bad

# ================= 第 0 步：估计步长 =================
print("📐 估计网格步长 dx / dy ...")

pf = pq.ParquetFile(FILE_IN)

xs_all = []
ys_all = []

for i, batch in enumerate(pf.iter_batches(batch_size=CHUNK, columns=[COL_X, COL_Y])):
    df = batch.to_pandas()
    xs_all.append(df[COL_X])
    ys_all.append(df[COL_Y])
    if i >= 10:  # 只用前 ~500 万行估计步长，够稳了
        break

xs_all = pd.concat(xs_all, ignore_index=True)
ys_all = pd.concat(ys_all, ignore_index=True)

dx_est = robust_step(xs_all)
dy_est = robust_step(ys_all)
dz_nominal = DZ_NOMINAL

print(f"dx ≈ {dx_est:.4f}, dy ≈ {dy_est:.4f}, dz ≈ {dz_nominal}")

x0 = xs_all.min()
y0 = ys_all.min()
z0 = None  # 稍后按 batch 动态定

# ================= 主循环 =================
print("🧾 开始 Parquet → Parquet 网格对齐 + 插值可疑度提取...")

writer = None
total = 0
bad_total = 0

pf = pq.ParquetFile(FILE_IN)

for i, batch in enumerate(pf.iter_batches(batch_size=CHUNK)):
    df = batch.to_pandas()

    if z0 is None:
        z0 = df[COL_Z].max()

    ix, X_snap, bad_x = snap_and_index(df[COL_X], x0, dx_est)
    iy, Y_snap, bad_y = snap_and_index(df[COL_Y], y0, dy_est)
    iz, Z_snap, bad_z = snap_and_index(df[COL_Z], z0, dz_nominal)

    bad_any = bad_x | bad_y | bad_z
    bad_total += bad_any.sum()

    # 丢弃严重偏网格的点（极少数）
    if bad_any.any():
        df = df[~bad_any].copy()
        ix, iy, iz = ix[~bad_any], iy[~bad_any], iz[~bad_any]
        X_snap, Y_snap, Z_snap = X_snap[~bad_any], Y_snap[~bad_any], Z_snap[~bad_any]

    # 更新吸附后的坐标（可选）
    df[COL_X], df[COL_Y], df[COL_Z] = X_snap, Y_snap, Z_snap

    df["ix"] = ix
    df["iy"] = iy
    df["iz"] = iz

    # ===== 核心：不去重，只计数 =====
    df["n_raw_in_cell"] = 1

    # ====== 写出（先不聚合，下一步再 groupby 全局聚）======
    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            FILE_OUT,
            table.schema,
            compression="zstd"
        )

    writer.write_table(table)
    total += len(df)

    if (i + 1) % 10 == 0:
        print(f"✅ batch {i+1} 完成，累计行数：{total:,}")

if writer is not None:
    writer.close()

print("✅ 网格吸附完成")
print("总行数：", total)
print("坐标异常丢弃行数：", bad_total)
print("📦 输出文件：", FILE_OUT)





# import pandas as pd
# import pyarrow.parquet as pq
# import pyarrow as pa
# import numpy as np
# import sys

# sys.stdout.reconfigure(encoding="utf-8")

# FILE_IN  = "属性_grid.parquet"
# FILE_OUT = "属性_grid_agg.parquet"

# CHUNK = 1_000_000

# print("🧮 开始格位级聚合（为极速反算做准备）...")

# dfs = []

# pf = pq.ParquetFile(FILE_IN)

# for i, batch in enumerate(pf.iter_batches(batch_size=CHUNK)):
#     df = batch.to_pandas()
#     dfs.append(df)

#     if (i + 1) % 10 == 0:
#         print(f"已读 batch {i+1}")

# print("🔗 合并并 groupby (ix,iy,iz) ...")
# df = pd.concat(dfs, ignore_index=True)

# # 只保留必要列
# keep_cols = ["ix", "iy", "iz", "地层"]
# df = df[keep_cols]

# # 对每个格位：取地层众数（最稳）
# df = (
#     df
#     .groupby(["ix", "iy", "iz"], as_index=False)
#     .agg({
#         "地层": lambda x: x.value_counts().idxmax()
#     })
# )

# table = pa.Table.from_pandas(df, preserve_index=False)
# pq.write_table(table, FILE_OUT, compression="zstd")

# print("✅ 聚合完成")
# print("唯一格位数：", len(df))
# print("📦 输出文件：", FILE_OUT)
