# import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq
# import xgboost as xgb
# import sys
# sys.stdout.reconfigure(encoding="utf-8")

# PARQUET = "feature_stack_3D_event_xylabel.parquet"
# MODEL_OUT = "xgb_xy_model.json"

# LABEL_COL = "label_xy"
# COORD_COLS = ["ix", "iy", "iz"]

# BATCH = 500_000
# U_SAMPLE_PER_BATCH = 50_000   # 控制 U 的比例（关键）

# # ========== 读 schema，找特征 ==========
# pf = pq.ParquetFile(PARQUET)  
# all_cols = pf.schema_arrow.names

# feat_cols = [c for c in all_cols if ("_cnt_k" in c or "_mean_k" in c)]
# print("特征维数：", len(feat_cols))

# # ========== 流式采样 ==========
# pos, weak, soft, u = [], [], [], []

# print("🔥 Streaming sampling for training...")

# for batch in pf.iter_batches(batch_size=BATCH):
#     df = batch.to_pandas()

#     pos.append(df[df[LABEL_COL] == 1])
#     weak.append(df[df[LABEL_COL] == 0])
#     soft.append(df[df[LABEL_COL] == -2])

#     ub = df[df[LABEL_COL] == -1]
#     if not ub.empty:
#         u.append(ub.sample(
#             n=min(len(ub), U_SAMPLE_PER_BATCH),
#             random_state=42
#         ))

# df_train = pd.concat(pos + weak + soft + u, ignore_index=True)
# print("训练集大小：", df_train.shape)
# print("label_xy 分布：")
# print(df_train[LABEL_COL].value_counts())

# # ========== 构造 X / y / w ==========
# X = df_train[feat_cols].astype(np.float32)
# y_raw = df_train[LABEL_COL].astype(np.int8)

# # 显式构造 y
# y = np.zeros(len(y_raw), dtype=np.int8)
# # y[y_raw >= -2] = 1      # 1, 0, -2 → positive
# # y[y_raw == -1] = 0      # -1 → negative
# y = (y_raw == 1).astype(np.int8)


# # w = np.ones(len(y), dtype=np.float32)
# # w[y_raw == 1]  = 9.0
# # w[y_raw == 0]  = 6.0
# # w[y_raw == -2] = 2.0
# # w[y_raw == -1] = 1.0

# # ========== 模型 ==========
# params = dict(
#     objective="binary:logistic",
#     eval_metric="aucpr",
#     tree_method="hist",
#     max_depth=6,
#     learning_rate=0.05,
#     subsample=0.85,
#     colsample_bytree=0.85,
#     n_estimators=500,
# )

# print("🏋️ 训练 XGBoost...")
# clf = xgb.XGBClassifier(**params, n_jobs=-1)
# clf.fit(X, y, sample_weight=w, verbose=False)

# clf.save_model(MODEL_OUT)
# print("✅ 模型已保存：", MODEL_OUT)















import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

# ===== 加载模型 =====
clf = xgb.XGBClassifier()
clf.load_model("xgb_xy_model.json")

# ===== 获取 importance =====
booster = clf.get_booster()
score = booster.get_score(importance_type='gain')

# ===== 转 DataFrame =====
df_imp = pd.DataFrame({
    "feature": list(score.keys()),
    "importance": list(score.values())
})




# ===== 归一化为百分比 =====
df_imp["importance_pct"] = df_imp["importance"] / df_imp["importance"].sum() * 100
# ===== 排序 =====
df_imp = df_imp.sort_values(by="importance_pct", ascending=False)

# ===== 只取前20（不然太乱）=====
df_plot = df_imp.head(11)
feature_name_map = {
    "Ag_o_mean_k(3, 21, 45)": "断层",
    "Cu_o_mean_k(5, 21, 45)": "地层",
    "stratum_flag_cnt_k(3, 45, 45)": "破碎带",
    "Ag_o_mean_k(1, 3, 3)": "磁化率",
    "fault_flag_cnt_k(3, 45, 45)": "岩性",
    "stratum_flag_cnt_k(1, 3, 3)": "矿体",
    "fault_flag_cnt_k(3, 21, 45)": "密度",
    "stratum_flag_cnt_k(5, 21, 21)": "电阻率",
    "stratum_flag_cnt_k(3, 21, 45)": "Ag",
    "else_flag_cnt_k(3, 45, 45)": "Cu",
    "stratum_flag_cnt_k(3, 21, 21)": "Mo"
}
df_plot["feature_cn"] = df_plot["feature"].map(feature_name_map)

# 防止有漏的
df_plot["feature_cn"] = df_plot["feature_cn"].fillna(df_plot["feature"])
# ===== 画图 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 6))
plt.barh(df_plot["feature_cn"], df_plot["importance_pct"])
plt.gca().invert_yaxis()

plt.xlabel("特征贡献度 (%)")
plt.title("五指山1:25000特征贡献 (XGBoost 收益)")

plt.tight_layout()
plt.show()

















# import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq
# import xgboost as xgb
# import sys
# sys.stdout.reconfigure(encoding="utf-8")

# PARQUET = "feature_stack_3D_event_xylabel.parquet"
# MODEL = "xgb_xy_model.json"
# OUT = "xy_proba.tsv"

# BATCH = 500_000

# # ========== 加载模型 ==========
# clf = xgb.XGBClassifier()
# clf.load_model(MODEL)

# pf = pq.ParquetFile(PARQUET)
# all_cols = pf.schema_arrow.names
# feat_cols = [c for c in all_cols if ("_cnt_k" in c or "_mean_k" in c)]

# print("🧮 开始全量流式预测...")

# with open(OUT, "w", encoding="utf-8-sig") as f:
#     f.write("ix\tiy\tiz\tproba\n")

#     for batch in pf.iter_batches(batch_size=BATCH):
#         df = batch.to_pandas()

#         Xb = df[feat_cols].astype(np.float32)
#         pb = clf.predict_proba(Xb)[:, 1].astype(np.float32)

#         out = pd.DataFrame({
#             "ix": df["ix"].to_numpy(),
#             "iy": df["iy"].to_numpy(),
#             "iz": df["iz"].to_numpy(),
#             "proba": pb
#         })

#         out.to_csv(f, sep="\t", index=False, header=False)

# print("✅ 预测完成：", OUT)








# import pandas as pd
# import numpy as np
# import pyarrow.parquet as pq
# import xgboost as xgb
# import sys
# sys.stdout.reconfigure(encoding="utf-8")

# # ✅ 用新的卷积输出
# PARQUET   = "feature_stack_2p5D_occ.parquet"
# MODEL_OUT = "pu_xgb_occ.json"
# PRED_OUT  = "pu_predictions_occ.tsv"

# label_col = "label"          # 你这个文件里还是原始 label
# coord_cols = ["ix","iy","iz"]

# pf = pq.ParquetFile(PARQUET)
# all_cols = pf.schema_arrow.names

# # ✅ 只用 occ 特征
# feat_cols = [c for c in all_cols if "_occ_k" in c]
# assert label_col in all_cols
# print("特征维数：", len(feat_cols), feat_cols)

# # ========= B) 流式采样 =========
# pos_rows, u_rows = [], []

# print("🔥 Streaming read + PU sampling...")

# for batch in pf.iter_batches(batch_size=500_000):
#     df = batch.to_pandas()

#     pos = df[df[label_col] == 1]     # 强正
#     u   = df[df[label_col] != 1]     # 其余都当 U（0 和 -1）

#     if not pos.empty:
#         pos_rows.append(pos)

#     if not u.empty:
#         u_rows.append(u.sample(n=min(len(u), 200_000), random_state=42))

# df_train = pd.concat(pos_rows + u_rows, ignore_index=True)
# print("训练集大小：", df_train.shape)
# print("label 分布：")
# print(df_train[label_col].value_counts())

# X = df_train[feat_cols].astype(np.float32)
# y = (df_train[label_col].to_numpy().astype(np.int8) == 1).astype(np.int8)  # ✅ 只有强正是 1
# print("PU y 分布：", np.unique(y, return_counts=True))

# # 权重：强调强正
# w = np.ones(len(y), dtype=np.float32)
# w[y == 1] = 8.0

# params = dict(
#     objective="binary:logistic",
#     eval_metric="aucpr",
#     tree_method="hist",
#     max_depth=5,
#     learning_rate=0.08,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     n_estimators=300,
# )

# print("🏋 训练模型...")
# clf = xgb.XGBClassifier(**params, n_jobs=-1)
# clf.fit(X, y, sample_weight=w, verbose=False)

# clf.save_model(MODEL_OUT)
# print("已保存模型：", MODEL_OUT)

# # ========= 全量流式预测 =========
# print("🧮 Streaming prediction...")
# pf_pred = pq.ParquetFile(PARQUET)

# with open(PRED_OUT, "w", encoding="utf-8-sig") as f:
#     f.write("ix\tiy\tiz\tproba\n")

#     for batch in pf_pred.iter_batches(batch_size=500_000):
#         df_b = batch.to_pandas()
#         X_b = df_b[feat_cols].astype(np.float32)
#         p_b = clf.predict_proba(X_b)[:, 1].astype(np.float32)

#         out = pd.DataFrame({
#             "ix": df_b["ix"].to_numpy(),
#             "iy": df_b["iy"].to_numpy(),
#             "iz": df_b["iz"].to_numpy(),
#             "proba": p_b
#         })
#         out.to_csv(f, sep="\t", index=False, header=False)

# print("✅ 完成预测：", PRED_OUT)
