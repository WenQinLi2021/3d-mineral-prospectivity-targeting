# import numpy as np
# import pandas as pd
# from scipy.ndimage import uniform_filter
# import pyarrow.parquet as pq
# import pyarrow as pa
# import sys
# sys.stdout.reconfigure(encoding="utf-8")

# # ================= 参数区 =================
# COL_X, COL_Y, COL_Z = "x", "y", "z"
# LABEL_COL = "label"

# EVENT_FEATS = ["Ag_o", "Al2O3_o", "Cu_o", "SiO2_o", "TS_o", "TFe_o"]

# DX, DY, DZ = 100.0, 100.0, 0.5
# REBIN_Z = 5

# # WINDOWS = [(1, 3, 3), (3, 21, 21), (3, 45, 45)]
# WINDOWS = [
#     (1, 3, 3),        # 点
#     (3, 21, 21),      # 局部连续（你之前最好的核心）
#     (3, 45, 45),      # 中尺度背景（别急着删它）
#     (7, 21, 21),      # Z 厚度（关键）
#     (3, 21, 45),      # 只加 1 个轻微非对称
# ]


# BLOCK_Z = 96
# #注意：窗口形状可控制
# IN_PATH  = "属性_with_label.parquet"
# OUT_PATH = "feature_stack_3D_event.parquet"

# BATCH_ROWS = 2_000_000
# EPS = 1e-6

# # ================= Step 1：扫描网格范围 =================
# print("🔍 扫描 Parquet 计算网格范围...")

# pf = pq.ParquetFile(IN_PATH)

# xmin, ymin, zmax = np.inf, np.inf, -np.inf
# xmax, ymax, zmin = -np.inf, -np.inf, np.inf

# for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=[COL_X, COL_Y, COL_Z]):
#     df = batch.to_pandas()
#     xmin = min(xmin, df[COL_X].min())
#     ymin = min(ymin, df[COL_Y].min())
#     zmax = max(zmax, df[COL_Z].max())

#     xmax = max(xmax, df[COL_X].max())
#     ymax = max(ymax, df[COL_Y].max())
#     zmin = min(zmin, df[COL_Z].min())

# x0, y0, z0 = float(xmin), float(ymin), float(zmax)

# NX = int(np.rint((xmax - x0) / DX)) + 1
# NY = int(np.rint((ymax - y0) / DY)) + 1
# NZ = int(np.rint((z0 - zmin) / DZ) // REBIN_Z) + 1

# print(f"Grid: NX={NX}, NY={NY}, NZ={NZ}")

# # ================= Step 2：Z-block 范围 =================
# max_kz = max(kz for (kz, _, _) in WINDOWS)
# overlap = max_kz // 2

# def layer_block_ranges(nz, B, ov):
#     z = 0
#     while z < nz:
#         z0b = max(0, z - ov)
#         z1b = min(nz, z + B + ov)
#         yield z, z0b, z1b
#         z += B

# # ================= Writer =================
# writer = None
# first_batch = True

# def write_block(df_block: pd.DataFrame):
#     global writer, first_batch
#     table = pa.Table.from_pandas(df_block, preserve_index=False)
#     if first_batch:
#         writer = pq.ParquetWriter(OUT_PATH, table.schema, compression="zstd")
#         first_batch = False
#     writer.write_table(table)

# # ================= Step 3：逐 Z-block 卷积 =================
# print("🚀 开始 Z-block 流式卷积...")

# for z_mid, z0_layer, z1_layer in layer_block_ranges(NZ, BLOCK_Z, overlap):
#     print(f"\n▶ Z block: core={z_mid} ~ {min(z_mid + BLOCK_Z, NZ)} (cache z[{z0_layer},{z1_layer}))")

#     # 缓存当前 block 所需行
#     rows_cache = []

#     pf2 = pq.ParquetFile(IN_PATH)
#     for batch in pf2.iter_batches(
#         batch_size=BATCH_ROWS,
#         columns=[COL_X, COL_Y, COL_Z, LABEL_COL] + EVENT_FEATS
#     ):
#         df = batch.to_pandas()

#         # iz：先按 DZ 取整，再 REBIN_Z
#         iz_raw = np.rint((z0 - df[COL_Z].to_numpy()) / DZ).astype(np.int32)
#         iz = (iz_raw // REBIN_Z).astype(np.int32)

#         m = (iz >= z0_layer) & (iz < z1_layer)
#         if m.any():
#             df = df.loc[m].copy()
#             df["iz"] = iz[m]
#             rows_cache.append(df)

#     if not rows_cache:
#         continue

#     df_block_all = pd.concat(rows_cache, ignore_index=True)
#     del rows_cache

#     # ix/iy
#     ix = np.rint((df_block_all[COL_X].to_numpy() - x0) / DX).astype(np.int32)
#     iy = np.rint((df_block_all[COL_Y].to_numpy() - y0) / DY).astype(np.int32)
#     iz = df_block_all["iz"].to_numpy().astype(np.int32)

#     df_block_all["ix"] = ix
#     df_block_all["iy"] = iy

#     # core 区域
#     core_mask = (iz >= z_mid) & (iz < min(z_mid + BLOCK_Z, NZ))
#     core_df = df_block_all.loc[core_mask]

#     ix_loc = core_df["ix"].to_numpy().astype(np.int32)
#     iy_loc = core_df["iy"].to_numpy().astype(np.int32)
#     iz_loc = core_df["iz"].to_numpy().astype(np.int32)
#     label_loc = core_df[LABEL_COL].to_numpy().astype(np.int8)

#     feat_dict = {
#         "ix": ix_loc,
#         "iy": iy_loc,
#         "iz": iz_loc,
#         LABEL_COL: label_loc
#     }

#     bz = int(z1_layer - z0_layer)
#     zsel = (iz_loc - z0_layer).astype(np.int32)

#     # 为本 block 预先算好索引（一次算，所有 feat 共用）
#     zz = (iz - z0_layer).astype(np.int32)
#     yy = iy.astype(np.int32)
#     xx = ix.astype(np.int32)

#     valid_xyz = (zz >= 0) & (zz < bz) & (yy >= 0) & (yy < NY) & (xx >= 0) & (xx < NX)

#     zzv_all = zz[valid_xyz]
#     yyv_all = yy[valid_xyz]
#     xxv_all = xx[valid_xyz]

#     # ================= 事件特征卷积 =================
#     for feat in EVENT_FEATS:
#         print(f"   · {feat}")

#         values = pd.to_numeric(df_block_all[feat], errors="coerce").fillna(0.0).to_numpy(np.float32)
#         valv_all = values[valid_xyz]

#         # 只对非零事件入格（极大加速 + 防止“零污染”）
#         nz = (valv_all != 0.0)
#         zzv = zzv_all[nz]
#         yyv = yyv_all[nz]
#         xxv = xxv_all[nz]
#         valv = valv_all[nz]

#         # 真实“体素内求和/计数”
#         sum_grid = np.zeros((bz, NY, NX), dtype=np.float32)
#         cnt_grid = np.zeros((bz, NY, NX), dtype=np.float32)

#         np.add.at(sum_grid, (zzv, yyv, xxv), valv)
#         np.add.at(cnt_grid, (zzv, yyv, xxv), 1.0)

#         # 逐窗口输出 cnt / mean
#         for (kz, ky, kx) in WINDOWS:
#             # 注意：uniform_filter 是均值，所以乘回体积 => 真计数/真求和
#             win_vol = float(kz * ky * kx)

#             cnt_k = uniform_filter(cnt_grid, size=(kz, ky, kx), mode="constant", cval=0.0) * win_vol
#             sum_k = uniform_filter(sum_grid, size=(kz, ky, kx), mode="constant", cval=0.0) * win_vol

#             mean_k = sum_k / np.maximum(cnt_k, 1.0)  # 用真实计数做分母

#             feat_dict[f"{feat}_cnt_k{(kz,ky,kx)}"]  = cnt_k[zsel, iy_loc, ix_loc].astype(np.float32)
#             feat_dict[f"{feat}_mean_k{(kz,ky,kx)}"] = mean_k[zsel, iy_loc, ix_loc].astype(np.float32)

#         # 释放大数组
#         del sum_grid, cnt_grid, cnt_k, sum_k, mean_k

#     # 写出
#     df_out = pd.DataFrame(feat_dict)

#     # 降低体积：float32
#     float_cols = df_out.select_dtypes(include="float").columns
#     df_out[float_cols] = df_out[float_cols].astype("float32")

#     write_block(df_out)

#     del df_out, df_block_all, feat_dict

# # 收尾
# if writer is not None:
#     writer.close()

# print(f"\n✅ 完成：{OUT_PATH}")




import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
import pyarrow.parquet as pq
import pyarrow as pa
import sys
sys.stdout.reconfigure(encoding="utf-8")

# ================= 参数区 =================
COL_X, COL_Y, COL_Z = "x", "y", "z"
LABEL_COL = "label"

# 原始化探事件特征
EVENT_FEATS = ["Ag_o", "Cu_o", "Mn_o"]

# 地层/断层列名（你现在就是这两个）
COL_STRATUM = "地层"
#COL_FAULT   = "断层"

DX, DY, DZ = 10.0, 10.0, 0.5
REBIN_Z = 5

WINDOWS = [
    (1, 3, 3),        # 点
    (3, 21, 21),      # 局部连续（你之前最好的核心）
    (3, 45, 45),      # 中尺度背景
    (7, 21, 21),      # Z 厚度（关键）
    (3, 21, 45),      # 只加 1 个轻微非对称
]

BLOCK_Z = 96

IN_PATH  = "属性_with_label.parquet"
OUT_PATH = "feature_stack_3D_event.parquet"

BATCH_ROWS = 2_000_000

# ========== 地层主类估计（抽样统计，不用全量） ==========
# 你也可以手动写死 MAIN_STRATUM=14；这里做成自动更稳
STRATUM_SAMPLE_LIMIT = 2_000_000

# ================= Step 1：扫描网格范围 + 估计主地层 =================
print("🔍 扫描 Parquet 计算网格范围 + 估计主地层...")

pf = pq.ParquetFile(IN_PATH)

xmin, ymin, zmax = np.inf, np.inf, -np.inf
xmax, ymax, zmin = -np.inf, -np.inf, np.inf

# 抽样统计地层频次（排除 0 / NaN）
stratum_counts = {}
sampled = 0
rng = np.random.default_rng(42)

for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=[COL_X, COL_Y, COL_Z, COL_STRATUM]):
    df = batch.to_pandas()

    xmin = min(xmin, df[COL_X].min())
    ymin = min(ymin, df[COL_Y].min())
    zmax = max(zmax, df[COL_Z].max())

    xmax = max(xmax, df[COL_X].max())
    ymax = max(ymax, df[COL_Y].max())
    zmin = min(zmin, df[COL_Z].min())

    # 抽样更新主地层统计
    if sampled < STRATUM_SAMPLE_LIMIT:
        s = pd.to_numeric(df[COL_STRATUM], errors="coerce").fillna(0).to_numpy(np.int32)
        s = s[s != 0]
        if s.size:
            # 只抽一部分，避免扫太慢
            take = min(s.size, STRATUM_SAMPLE_LIMIT - sampled)
            if take < s.size:
                idx = rng.choice(s.size, size=take, replace=False)
                s = s[idx]
            sampled += s.size
            u, c = np.unique(s, return_counts=True)
            for k, v in zip(u.tolist(), c.tolist()):
                stratum_counts[k] = stratum_counts.get(k, 0) + int(v)

x0, y0, z0 = float(xmin), float(ymin), float(zmax)

NX = int(np.rint((xmax - x0) / DX)) + 1
NY = int(np.rint((ymax - y0) / DY)) + 1
NZ = int(np.rint((z0 - zmin) / DZ) // REBIN_Z) + 1

# 主地层
if stratum_counts:
    MAIN_STRATUM = max(stratum_counts.items(), key=lambda kv: kv[1])[0]
else:
    MAIN_STRATUM = 0

print(f"Grid: NX={NX}, NY={NY}, NZ={NZ}")
print(f"✅ MAIN_STRATUM（抽样众数）= {MAIN_STRATUM} | sampled={sampled:,}")

# ================= Step 2：Z-block 范围 =================
max_kz = max(kz for (kz, _, _) in WINDOWS)
overlap = max_kz // 2

def layer_block_ranges(nz, B, ov):
    z = 0
    while z < nz:
        z0b = max(0, z - ov)
        z1b = min(nz, z + B + ov)
        yield z, z0b, z1b
        z += B

# ================= Writer =================
writer = None
first_batch = True

def write_block(df_block: pd.DataFrame):
    global writer, first_batch
    table = pa.Table.from_pandas(df_block, preserve_index=False)
    if first_batch:
        writer = pq.ParquetWriter(OUT_PATH, table.schema, compression="zstd")
        first_batch = False
    writer.write_table(table)

# ================= Step 3：逐 Z-block 卷积 =================
print("🚀 开始 Z-block 流式卷积...")

# 我们把结构事件特征追加到 EVENT_FEATS 里（卷积逻辑复用）
EVENT_FEATS_ALL = EVENT_FEATS + ["stratum_flag"]#, "fault_flag"]

for z_mid, z0_layer, z1_layer in layer_block_ranges(NZ, BLOCK_Z, overlap):
    print(f"\n▶ Z block: core={z_mid} ~ {min(z_mid + BLOCK_Z, NZ)} (cache z[{z0_layer},{z1_layer}))")

    rows_cache = []

    pf2 = pq.ParquetFile(IN_PATH)
    for batch in pf2.iter_batches(
        batch_size=BATCH_ROWS,
        columns=[COL_X, COL_Y, COL_Z, LABEL_COL, COL_STRATUM]+ EVENT_FEATS#, COL_FAULT] + EVENT_FEATS
    ):
        df = batch.to_pandas()

        # iz：先按 DZ 取整，再 REBIN_Z
        iz_raw = np.rint((z0 - df[COL_Z].to_numpy()) / DZ).astype(np.int32)
        iz = (iz_raw // REBIN_Z).astype(np.int32)

        m = (iz >= z0_layer) & (iz < z1_layer)
        if m.any():
            df = df.loc[m].copy()
            df["iz"] = iz[m]
            rows_cache.append(df)

    if not rows_cache:
        continue

    df_block_all = pd.concat(rows_cache, ignore_index=True)
    del rows_cache

    # ix/iy/iz
    ix = np.rint((df_block_all[COL_X].to_numpy() - x0) / DX).astype(np.int32)
    iy = np.rint((df_block_all[COL_Y].to_numpy() - y0) / DY).astype(np.int32)
    iz = df_block_all["iz"].to_numpy().astype(np.int32)

    df_block_all["ix"] = ix
    df_block_all["iy"] = iy

    # ========== 结构事件特征：stratum_flag / fault_flag ==========
    # stratum_flag：非 0 且非主地层 => 认为是“层位变化/接触/异常层”
    st = pd.to_numeric(df_block_all[COL_STRATUM], errors="coerce").fillna(0).to_numpy(np.int32)
    #fl = pd.to_numeric(df_block_all[COL_FAULT], errors="coerce").fillna(0).to_numpy(np.int32)

    df_block_all["stratum_flag"] = ((st != 0) & (st != MAIN_STRATUM)).astype(np.float32)
    #df_block_all["fault_flag"]   = (fl != 0).astype(np.float32)

    # core 区域
    core_mask = (iz >= z_mid) & (iz < min(z_mid + BLOCK_Z, NZ))
    core_df = df_block_all.loc[core_mask]

    ix_loc = core_df["ix"].to_numpy().astype(np.int32)
    iy_loc = core_df["iy"].to_numpy().astype(np.int32)
    iz_loc = core_df["iz"].to_numpy().astype(np.int32)
    label_loc = core_df[LABEL_COL].to_numpy().astype(np.int8)

    feat_dict = {
        "ix": ix_loc,
        "iy": iy_loc,
        "iz": iz_loc,
        LABEL_COL: label_loc
    }

    bz = int(z1_layer - z0_layer)
    zsel = (iz_loc - z0_layer).astype(np.int32)

    # 为本 block 预先算好索引（一次算，所有 feat 共用）
    zz = (iz - z0_layer).astype(np.int32)
    yy = iy.astype(np.int32)
    xx = ix.astype(np.int32)

    valid_xyz = (zz >= 0) & (zz < bz) & (yy >= 0) & (yy < NY) & (xx >= 0) & (xx < NX)

    zzv_all = zz[valid_xyz]
    yyv_all = yy[valid_xyz]
    xxv_all = xx[valid_xyz]

    # ================= 事件特征卷积 =================
    for feat in EVENT_FEATS_ALL:
        print(f"   · {feat}")

        values = pd.to_numeric(df_block_all[feat], errors="coerce").fillna(0.0).to_numpy(np.float32)
        valv_all = values[valid_xyz]

        # 只对非零事件入格（极大加速 + 防止“零污染”）
        nz_mask = (valv_all != 0.0)
        if not nz_mask.any():
            # 这个特征在该 block 完全为 0，直接填 0 输出
            for (kz, ky, kx) in WINDOWS:
                feat_dict[f"{feat}_cnt_k{(kz,ky,kx)}"]  = np.zeros(len(ix_loc), dtype=np.float32)
                feat_dict[f"{feat}_mean_k{(kz,ky,kx)}"] = np.zeros(len(ix_loc), dtype=np.float32)
            continue

        zzv = zzv_all[nz_mask]
        yyv = yyv_all[nz_mask]
        xxv = xxv_all[nz_mask]
        valv = valv_all[nz_mask]

        sum_grid = np.zeros((bz, NY, NX), dtype=np.float32)
        cnt_grid = np.zeros((bz, NY, NX), dtype=np.float32)

        np.add.at(sum_grid, (zzv, yyv, xxv), valv)
        np.add.at(cnt_grid, (zzv, yyv, xxv), 1.0)

        for (kz, ky, kx) in WINDOWS:
            win_vol = float(kz * ky * kx)

            cnt_k = uniform_filter(cnt_grid, size=(kz, ky, kx), mode="constant", cval=0.0) * win_vol
            sum_k = uniform_filter(sum_grid, size=(kz, ky, kx), mode="constant", cval=0.0) * win_vol

            mean_k = sum_k / np.maximum(cnt_k, 1.0)

            feat_dict[f"{feat}_cnt_k{(kz,ky,kx)}"]  = cnt_k[zsel, iy_loc, ix_loc].astype(np.float32)
            feat_dict[f"{feat}_mean_k{(kz,ky,kx)}"] = mean_k[zsel, iy_loc, ix_loc].astype(np.float32)

        del sum_grid, cnt_grid, cnt_k, sum_k, mean_k

    df_out = pd.DataFrame(feat_dict)

    # 降低体积：float32
    float_cols = df_out.select_dtypes(include="float").columns
    df_out[float_cols] = df_out[float_cols].astype("float32")

    write_block(df_out)

    del df_out, df_block_all, feat_dict

if writer is not None:
    writer.close()

print(f"\n✅ 完成：{OUT_PATH}")





# import pyarrow.parquet as pq
# pf = pq.ParquetFile("feature_stack_2p5D_occ.parquet")
# print([c for c in pf.schema_arrow.names if "_occ_k" in c])



# import pyarrow.parquet as pq

# pf = pq.ParquetFile("feature_stack_3D_event.parquet")
# cols = pf.schema_arrow.names

# # 看看 TS / Ag 的 cnt 列到底叫什么
# for c in cols:
#     if "TS_o_cnt" in c or "Ag_o_cnt" in c:
#         print(c)
