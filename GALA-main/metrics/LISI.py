from scib.metrics.lisi import lisi_graph
import numpy as np

def calculate_lisi(adata_post, emb_key, label_key, batch_key, verbose=True):

    # 调试：检查批次和细胞类型的多样性
    n_batch = adata_post.obs[batch_key].nunique()
    n_celltype = adata_post.obs[label_key].nunique()
    if verbose:
        print(f"Number of batches: {n_batch}")
        print(f"Number of cell types: {n_celltype}")

    if n_batch == 1 or n_celltype == 1:
        print("Warning: Only one batch or cell type detected. LISI values will be 1.")
        return 0.0, 1.0, 1.0

    # 调试：检查嵌入矩阵
    embedding = adata_post.obsm[emb_key]
    if verbose:
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding min: {embedding.min()}, max: {embedding.max()}")

    try:
        ilisi_score, clisi_score = lisi_graph(adata_post, batch_key=batch_key, label_key=label_key, type_="graph")
        if verbose:
            print(f"iLISI (from lisi_graph): {ilisi_score:.4f}, cLISI (from lisi_graph): {clisi_score:.4f}")
    except Exception as e:
        print(f"Error in lisi_graph: {e}")
        return np.nan, np.nan, np.nan

    # 计算 F1 LISI
    if ilisi_score + clisi_score != 0:
        f1_lisi = (2 * ilisi_score * clisi_score) / (ilisi_score + clisi_score)
    else:
        f1_lisi = np.nan

    if verbose:
        print(f"F1 LISI: {f1_lisi:.4f}, iLISI: {ilisi_score:.4f}, cLISI: {clisi_score:.4f}")

    return f1_lisi, ilisi_score, clisi_score