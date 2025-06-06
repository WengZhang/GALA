import scipy
import scanpy as sc
import numpy as np

def data_preprocessing(adata):
    """Function used to preprocess our data with batch effect
    """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    # 确保批次信息为 'category' 类型
    if 'batch' in adata.obs:
        adata.obs['batch'] = adata.obs['batch'].astype('category')
    else:
        raise ValueError("Batch key not found in adata.obs")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key='batch')
    adata = adata[:, adata.var['highly_variable']]
    # check if data is in sparse format
    if isinstance(adata.X, scipy.sparse.csr.csr_matrix): #判断adata.x是否是一个CSR格式的稀疏矩阵
        adata_new = sc.AnnData(adata.X.todense())
        adata_new.obs = adata.obs.copy()
        adata_new.obs_names = adata.obs_names
        adata_new.var_names = adata.var_names
        adata_new.obs_names.name = 'CellID'
        adata_new.var_names.name = 'Gene'
        del adata
        adata = adata_new
    return adata

# def data_preprocessing(adata):
#     """Function used to preprocess our data with batch effect"""
#     # 过滤细胞和基因
#     sc.pp.filter_cells(adata, min_genes=200)
#     sc.pp.filter_genes(adata, min_cells=3)
#
#     # 标准化和对数化
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#
#     # 确保批次信息为 'category' 类型
#     if 'batch' in adata.obs:
#         adata.obs['batch'] = adata.obs['batch'].astype('category')
#     else:
#         raise ValueError("Batch key not found in adata.obs")
#
#     # 选择高变基因（不按批次选择）
#     sc.pp.highly_variable_genes(adata, n_top_genes=2000)
#     adata = adata[:, adata.var['highly_variable']]
#
#     # 标准化（零均值，单位方差）
#     sc.pp.scale(adata, max_value=10)
#
#     return adata