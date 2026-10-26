# import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq
# import pyarrow as pa
# from scipy.ndimage import binary_dilation
# import sys
# sys.stdout.reconfigure(encoding="utf-8")
# IN_PARQUET  = "feature_stack_3D_event.parquet"
# OUT_PARQUET = "feature_stack_3D_event_xylabel.parquet"

# LABEL_COL = "label"
# NEW_LABEL = "label_xy"

# BATCH = 500_000
# RADIUS = 4   # XY 扩散半径（格数）

# pf = pq.ParquetFile(IN_PARQUET)
# writer = None

# kernel = np.ones((2*RADIUS+1, 2*RADIUS+1), dtype=bool)

# for batch in pf.iter_batches(batch_size=BATCH):
#     df = batch.to_pandas()

#     label_xy = df[LABEL_COL].to_numpy().copy()

#     for iz in np.unique(df["iz"]):
#         m = (df["iz"] == iz)
#         if not m.any():
#             continue

#         sub = df.loc[m]
#         ix = sub["ix"].to_numpy()
#         iy = sub["iy"].to_numpy()
#         lab = sub[LABEL_COL].to_numpy()

#         # 只看 label >= 0 的点
#         pos = (lab >= 0)
#         if not pos.any():
#             continue

#         nx = ix.max() + 1
#         ny = iy.max() + 1

#         grid = np.zeros((ny, nx), dtype=bool)
#         grid[iy[pos], ix[pos]] = True

#         grid_dil = binary_dilation(grid, structure=kernel)

#         # 把被扩散到、且原来是 -1 的点 → -2
#         for i, (x, y) in enumerate(zip(ix, iy)):
#             if lab[i] == -1 and grid_dil[y, x]:
#                 label_xy[m.index[i]] = -2

#     df[NEW_LABEL] = label_xy

#     table = pa.Table.from_pandas(df, preserve_index=False)

#     if writer is None:
#         writer = pq.ParquetWriter(
#             OUT_PARQUET,
#             table.schema,
#             compression="zstd"
#         )

#     writer.write_table(table)

# if writer is not None:
#     writer.close()

# print("✅ 已生成带 XY 扩散标签的特征文件：", OUT_PARQUET)





import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from scipy.ndimage import binary_dilation
import sys
sys.stdout.reconfigure(encoding="utf-8")

IN_PARQUET  = "feature_stack_3D_event.parquet"
OUT_PARQUET = "feature_stack_3D_event_xylabel.parquet"

LABEL_COL = "label"
NEW_LABEL = "label_xy"

BATCH = 500_000
RADIUS = 4   # XY 扩散半径（格）

# ========= 圆形 XY 扩散核（非常重要） =========
yy, xx = np.ogrid[-RADIUS:RADIUS+1, -RADIUS:RADIUS+1]
kernel_xy = (xx*xx + yy*yy) <= (RADIUS*RADIUS)

pf = pq.ParquetFile(IN_PARQUET)
writer = None

print("🚀 开始生成 XY + Z 双软扩散标签 (-2 / -3)...")

for batch in pf.iter_batches(batch_size=BATCH):
    df = batch.to_pandas()

    label_xy = df[LABEL_COL].to_numpy().copy()

    iz_all = df["iz"].to_numpy()
    ix_all = df["ix"].to_numpy()
    iy_all = df["iy"].to_numpy()

    unique_iz = np.unique(iz_all)

    # ========= 逐 iz 处理 =========
    for iz in unique_iz:
        m = (iz_all == iz)
        if not m.any():
            continue

        ix = ix_all[m]
        iy = iy_all[m]
        lab = label_xy[m]   # 注意：用 label_xy，允许叠加

        pos = (lab >= 0)
        if not pos.any():
            continue

        nx = ix.max() + 1
        ny = iy.max() + 1

        grid = np.zeros((ny, nx), dtype=bool)
        grid[iy[pos], ix[pos]] = True

        grid_dil = binary_dilation(grid, structure=kernel_xy)

        # ===== (A) 同层 XY 扩散 → -2 =====
        for i, (x, y) in enumerate(zip(ix, iy)):
            if lab[i] == -1 and grid_dil[y, x]:
                label_xy[np.where(m)[0][i]] = -2

        # ===== (B) 邻层 Z 扩散 → -3 =====
        for iz2 in (iz - 1, iz + 1):
            m2 = (iz_all == iz2)
            if not m2.any():
                continue

            ix2 = ix_all[m2]
            iy2 = iy_all[m2]
            lab2 = label_xy[m2]

            for i2, (x2, y2) in enumerate(zip(ix2, iy2)):
                if lab2[i2] != -1:
                    continue
                if 0 <= y2 < ny and 0 <= x2 < nx:
                    if grid_dil[y2, x2]:
                        label_xy[np.where(m2)[0][i2]] = -3

    df[NEW_LABEL] = label_xy

    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            OUT_PARQUET,
            table.schema,
            compression="zstd"
        )

    writer.write_table(table)

if writer is not None:
    writer.close()

print("✅ 已生成带 XY + Z 软扩散标签的特征文件：", OUT_PARQUET)






# import pyarrow.parquet as pq
# import pandas as pd

# PARQUET = "feature_stack_3D_event_xylabel.parquet"

# pf = pq.ParquetFile(PARQUET)

# counts = {}

# for batch in pf.iter_batches(batch_size=500_000, columns=["label_xy"]):
#     df = batch.to_pandas()
#     vc = df["label_xy"].value_counts()

#     for k, v in vc.items():
#         counts[k] = counts.get(k, 0) + int(v)

# print("label_xy 分布：")
# for k in sorted(counts):
#     print(f"{k:>3}: {counts[k]:,}")
