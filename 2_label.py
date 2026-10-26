# # # ===================  方案3 ==============#
# import pandas as pd
# import numpy as np
# import pyarrow.parquet as pq
# import pyarrow as pa
# import sys

# sys.stdout.reconfigure(encoding="utf-8")

# INPUT_PARQUET  = "属性.parquet"
# OUTPUT_PARQUET = "属性_with_label.parquet"

# COL_PB = "Pb_o"
# #COL_ZN = "Zn_o"
# TARGET_COL = [COL_PB]#, COL_ZN]

# CHUNK = 500_000

# # ======== 第 1 遍：分别估计 Pb / Zn 的 25% 分位数（仅 >0）========
# print("📊 估计 Pb/Zn 的 25% 分位数（仅 >0，蓄水池抽样）...")

# pf = pq.ParquetFile(INPUT_PARQUET)

# SAMPLE_LIMIT = 2_000_000
# rng = np.random.default_rng(42)

# pb_sample = []
# #zn_sample = []

# def reservoir_push(arr_1d, sample_list):
#     # arr_1d: 1D numpy array
#     for val in arr_1d:
#         if len(sample_list) < SAMPLE_LIMIT:
#             sample_list.append(val)
#         else:
#             j = rng.integers(0, len(sample_list) + 1)
#             if j < SAMPLE_LIMIT:
#                 sample_list[j] = val

# for batch in pf.iter_batches(batch_size=CHUNK, columns=TARGET_COL):
#     df = batch.to_pandas()

#     pb = pd.to_numeric(df[COL_PB], errors="coerce").fillna(0.0).to_numpy()
#     #zn = pd.to_numeric(df[COL_ZN], errors="coerce").fillna(0.0).to_numpy()

#     pb_pos = pb[pb > 0]
#     #zn_pos = zn[zn > 0]

#     if pb_pos.size:
#         reservoir_push(pb_pos, pb_sample)
#     #if zn_pos.size:
#         #reservoir_push(zn_pos, zn_sample)

# pb_q75 = float(np.quantile(np.array(pb_sample), 0.75)) if pb_sample else 0.0
# #zn_q75 = float(np.quantile(np.array(zn_sample), 0.75)) if zn_sample else 0.0

# print(f"✅ Pb 25% 分位数（近似，仅>0） = {pb_q75:.6g} | 抽样={len(pb_sample):,}")
# #print(f"✅ Zn 25% 分位数（近似，仅>0） = {zn_q75:.6g} | 抽样={len(zn_sample):,}")

# # ======== 第 2 遍：按 OR 规则打三态标签 ========
# print("🧾 开始打标签（OR：Pb 或 Zn 进入前25% 即 label=1）...")

# writer = None
# total = 0

# for i, batch in enumerate(pf.iter_batches(batch_size=CHUNK)):
#     df = batch.to_pandas()

#     pb = pd.to_numeric(df[COL_PB], errors="coerce").fillna(0.0).to_numpy()
#     #zn = pd.to_numeric(df[COL_ZN], errors="coerce").fillna(0.0).to_numpy()

#     label = np.full(len(df), -1, dtype=np.int8)

#     has_any = (pb > 0) #| (zn > 0)
#     in_front25 = ((pb > 0) & (pb >= pb_q75)) #| ((zn > 0) & (zn >= zn_q75))

#     label[has_any] = 0
#     label[in_front25] = 1

#     df["label"] = label

#     table = pa.Table.from_pandas(df, preserve_index=False)

#     if writer is None:
#         writer = pq.ParquetWriter(
#             OUTPUT_PARQUET,
#             table.schema,
#             compression="zstd"
#         )

#     writer.write_table(table)

#     total += len(df)
#     if (i + 1) % 10 == 0:
#         print(f"✅ batch {i+1} 完成，累计行数：{total:,}")

# if writer is not None:
#     writer.close()

# print("✅ 完成，输出：", OUTPUT_PARQUET)






# import pyarrow.parquet as pq
# import pandas as pd
# import sys
# sys.stdout.reconfigure(encoding="utf-8")
# FILE = "pu_predictions.parquet"

# print("📥 打开 Parquet 文件...")
# pf = pq.ParquetFile(FILE)

# print("🧱 schema:")
# print(pf.schema)

# print("\n📊 基本信息:")
# print("  行数:", pf.metadata.num_rows)
# print("  列数:", pf.metadata.num_columns)
# print("  列名:", [pf.schema.names[i] for i in range(pf.metadata.num_columns)])



import pyarrow.parquet as pq
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")

FILE = "属性_with_label.parquet"
CHUNK = 1_000_000

cnt = {1: 0, 0: 0, -1: 0}
total = 0

print("📊 扫描 label 分布...")

pf = pq.ParquetFile(FILE)

for batch in pf.iter_batches(batch_size=CHUNK, columns=["label"]):
    s = batch.to_pandas()["label"]
    total += len(s)
    for k in cnt:
        cnt[k] += (s == k).sum()

print("\n📊 label 统计结果")
print("────────────────────────")
print("总行数:", total)
print("label = 1 :", cnt[1])
print("label = 0 :", cnt[0])
print("label = -1:", cnt[-1])

print("\n📐 占比:")
for k in cnt:
    print(f"  label = {k}: {cnt[k] / total * 100:.4f}%")
