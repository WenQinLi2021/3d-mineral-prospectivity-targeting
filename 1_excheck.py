# with open('属性.txt', 'r', encoding='gb2312') as f_in, \
#      open('属性_utf8.txt', 'w', encoding='utf-8-sig') as f_out: 
#     for line in f_in: f_out.write(line)







# import sys
# sys.stdout.reconfigure(encoding="utf-8")
# in_file  = "属性_utf8.txt"
# out_file = "属性_fixed.csv"

# with open(in_file, "r", encoding="utf-8-sig") as fin, \
#      open(out_file, "w", encoding="utf-8-sig") as fout:

#     for i, line in enumerate(fin):
#         if i > 0:
#             # 只修表头行：TAB -> 逗号
#             line = line.replace(",", "\t")
#         fout.write(line)

# print("✅ 修复完成：表头已从 TAB 改为逗号")









# import chardet
# from collections import Counter
# import sys
# sys.stdout.reconfigure(encoding="utf-8")

# FILE = "属性_fixed.csv"
# SAMPLE_LINES = 5000

# print("🔍 Step 0.1: 检测编码...")
# with open(FILE, "rb") as f:
#     raw = f.read(200_000)

# enc = chardet.detect(raw)
# print("  ➤ 编码猜测:", enc)

# print("\n🔍 Step 0.2: 读取前几行做分隔符与列数检测...")

# candidate_seps = ["\t", ",", ";", "|"]
# sep_scores = {}

# with open(FILE, "r", encoding=enc["encoding"], errors="replace") as f:
#     lines = [next(f) for _ in range(50)]

# for sep in candidate_seps:
#     lens = [len(line.strip().split(sep)) for line in lines]
#     sep_scores[sep] = Counter(lens).most_common(3)

# print("  ➤ 各分隔符下列数分布：")
# for k, v in sep_scores.items():
#     print(f"    '{k}':", v)

# best_sep = max(sep_scores, key=lambda k: sep_scores[k][0][0])
# print("\n  ✅ 推荐分隔符:", repr(best_sep))

# print("\n🔍 Step 0.3: 全文件列数一致性抽检...")

# col_counts = Counter()
# bad_lines = []

# with open(FILE, "r", encoding=enc["encoding"], errors="replace") as f:
#     for i, line in enumerate(f):
#         if i >= SAMPLE_LINES:
#             break
#         n = len(line.rstrip("\n").split(best_sep))
#         col_counts[n] += 1
#         if i < 5:
#             print(f"  行 {i+1}:", line[:120])
#         if i > 0 and n != list(col_counts.keys())[0]:
#             bad_lines.append((i+1, n))

# print("\n  ➤ 列数统计:", col_counts)
# print("  ➤ 可疑行数:", len(bad_lines))
# if bad_lines[:5]:
#     print("  前 5 个异常行:", bad_lines[:5])





# #====================格式======================
# 456
# #==== 预处理，为的是搞清楚标签的特征是什么 =====
# import chardet
# from collections import Counter
# import sys
# import pandas as pd
# sys.stdout.reconfigure(encoding="utf-8")

# FILE = "属性_fixed.txt"
# TARGET_COL = "Pb_o"
# CHUNK = 1_000_000

# total = 0
# cnt_nonzero = 0
# cnt_nonneg1 = 0
# sum_val = 0.0
# min_val = float("inf")
# max_val = float("-inf")

# zero_count = 0
# neg1_count = 0

# print("📥 流式扫描目标列:", TARGET_COL)

# for chunk in pd.read_csv(FILE, sep="\t", encoding="utf-8-sig", chunksize=CHUNK):
#     col = chunk[TARGET_COL]

#     total += len(col)

#     zero_count += (col == 0).sum()
#     neg1_count += (col == -1).sum()

#     mask_nonzero = col != 0
#     mask_nonneg1 = col != -1
#     mask_valid = (col != 0) & (col != -1)

#     cnt_nonzero += mask_nonzero.sum()
#     cnt_nonneg1 += mask_nonneg1.sum()

#     if mask_valid.any():
#         vals = col[mask_valid]
#         sum_val += vals.sum()
#         min_val = min(min_val, vals.min())
#         max_val = max(max_val, vals.max())

# print("\n📊 目标列统计结果")
# print("────────────────────────")
# print("总行数:", total)
# print("== 0 的条数:", zero_count)
# print("== -1 的条数:", neg1_count)
# print("!= 0 的条数:", cnt_nonzero)
# print("!= -1 的条数:", cnt_nonneg1)

# valid_count = total - zero_count - neg1_count
# print("真正有效值条数 (非 0 且非 -1):", valid_count)

# if valid_count > 0:
#     mean_val = sum_val / valid_count
#     print("最小值:", min_val)
#     print("最大值:", max_val)
#     print("平均值:", mean_val)
# else:
#     print("⚠ 没有任何有效值（全是 0 或 -1）")


import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ================= 配置区 =================
FILE_IN  = "属性_fixed.txt"
FILE_OUT = "属性.parquet"

CHUNK = 500_000

DTYPES = {
    "index": "int64",
    "x": "float32",
    "y": "float32",
    "z": "float32",
    "地层": "int64",
    "Ag_o": "float32",
    "Cu_o": "float32",
    "Pb_o": "float32",
    "Mn_o": "float32",
}

# ================= 主循环 =================
print("📥 开始 TXT → Parquet...")

writer = None
total = 0

for i, chunk in enumerate(pd.read_csv(
    FILE_IN,
    sep="\t",
    encoding="utf-8-sig",
    chunksize=CHUNK
)):
    # 强制列类型（防止 object / 混类型）
    for col, dt in DTYPES.items():
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").astype(dt)

    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            FILE_OUT,
            table.schema,
            compression="zstd"
        )

    writer.write_table(table)
    total += len(chunk)

    if (i + 1) % 10 == 0:
        print(f"✅ batch {i+1} 完成，累计行数：{total:,}")

if writer is not None:
    writer.close()

print("✅ 转换完成，总行数：", total)
print("📦 输出文件：", FILE_OUT)


# import pyarrow.parquet as pq
# import sys
# sys.stdout.reconfigure(encoding="utf-8")
# pf = pq.ParquetFile("属性.parquet")
# batch = next(pf.iter_batches(batch_size=20, columns=["Tfe"]))
# df = batch.to_pandas()

# print(df)
# print("列名：")
# print(pf.schema_arrow.names)

# print("\n完整 schema：")
# print(pf.schema_arrow)


