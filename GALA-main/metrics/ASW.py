from sklearn.metrics import silhouette_score
import numpy as np
import random

def calculate_ASW(adata, labels=None, total_cells=None, percent_extract=0.8, batch_key='batch', celltype_key=None,
                  verbose=True):
    random.seed(0)
    np.random.seed(0)

    # 存储每次迭代的结果
    asw_f1 = []
    asw_b = []
    asw_c = []

    for i in range(20):
        # 随机提取数据子集
        rand_idx = np.random.choice(adata.obs_names, size=int(len(adata.obs_names) * percent_extract), replace=False)
        adata_sub = adata[rand_idx]

        if celltype_key is None:
            # 计算 1-bASW（批次效应），直接使用原始值
            try:
                asw_batch = silhouette_score(adata_sub.obsm['X_pca'], adata_sub.obs[batch_key])
                asw_b.append(1 - asw_batch)  # 直接使用 1 - 原始值
            except ValueError:
                asw_b.append(np.nan)  # 如果样本不足，设为 NaN
            asw_c.append(np.nan)  # 如果没有细胞类型，cASW 设为 NaN
            asw_f1.append(np.nan)
        else:
            # 计算 cASW（细胞类型效应），直接使用原始值
            try:
                asw_celltype = silhouette_score(adata_sub.obsm['X_pca'], adata_sub.obs[celltype_key])
                asw_c.append(asw_celltype)  # 直接使用原始值
            except ValueError:
                asw_c.append(np.nan)

            # 计算 1-bASW
            temp = 0
            total_cells_used = 0  # 动态计算实际参与计算的细胞数量
            for label in labels:
                adata_sub_c = adata_sub[adata_sub.obs[celltype_key] == label]
                if adata_sub_c.shape[0] < 10 or len(set(adata_sub_c.obs[batch_key])) == 1:
                    continue  # 跳过样本不足或批次单一的情况
                try:
                    asw_batch = silhouette_score(adata_sub_c.obsm['X_pca'], adata_sub_c.obs[batch_key])
                    temp += (1 - asw_batch) * adata_sub_c.shape[0]  # 直接使用 1 - 原始值
                    total_cells_used += adata_sub_c.shape[0]
                except ValueError:
                    continue  # 如果样本不足，跳过
            if total_cells_used > 0:  # 防止除以零
                temp /= total_cells_used
            asw_b.append(temp)

            # 计算 F1 ASW
            if temp + asw_celltype != 0:
                asw_fscore = (2 * temp * asw_celltype) / (temp + asw_celltype)
            else:
                asw_fscore = np.nan
            asw_f1.append(asw_fscore)

    # 计算所有迭代结果的均值，忽略 NaN 值
    asw_c = np.nanmean(asw_c)
    asw_b = np.nanmean(asw_b)
    asw_f1 = np.nanmean(asw_f1)

    if verbose:
        print(f'cASW: {asw_c:.4f}, 1-bASW: {asw_b:.4f}, F1 ASW: {asw_f1:.4f}')

    return asw_c, asw_b, asw_f1