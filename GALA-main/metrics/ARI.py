import numpy as np
from sklearn.metrics import adjusted_rand_score


def calculate_ARI_celltype(true_labels, pred_labels):
    """
    Calculate ARI for cell type purity.

    Args:
        true_labels: True cell type labels.
        pred_labels: Predicted cluster labels.

    Returns:
        ari_celltype: Adjusted Rand Index for cell type purity, range [0, 1].
    """
    ari_celltype = adjusted_rand_score(true_labels, pred_labels)
    return ari_celltype


def calculate_ARI_batch(batch_labels, pred_labels):
    """
    Calculate 1 - ARI for batch mixing (as per the PDF definition).

    Args:
        batch_labels: True batch labels.
        pred_labels: Predicted cluster labels.

    Returns:
        batch_score: 1 - ARI_batch, range [0, 1], higher is better.
    """
    ari_batch = adjusted_rand_score(batch_labels, pred_labels)
    batch_score = 1 - ari_batch  # This matches the PDF's 1 - ARI_batch
    return batch_score


def calculate_ARI(adata):
    """
    Function to compute 1 - ARI_batch, ARI_celltype, and F1_ARI for batch mixing and cell type purity.

    Args:
        adata: AnnData object containing cell type, batch, and predicted cluster labels.

    Returns:
        ari_celltype: ARI for cell type purity, range [0, 1].
        batch_score: 1 - ARI_batch for batch mixing, range [0, 1].
        f1_ari: F1 score combining batch mixing and cell type purity, range [0, 1].
    """
    # Extract labels from adata
    true_labels = adata.obs['cell_type']  # True cell type labels
    pred_labels = adata.obs['louvain']  # Predicted cluster labels (from Louvain clustering)
    batch_labels = adata.obs['batch']  # True batch labels

    # Calculate ARI for cell type purity
    ari_celltype = calculate_ARI_celltype(true_labels, pred_labels)

    # Calculate 1 - ARI_batch for batch mixing
    batch_score = calculate_ARI_batch(batch_labels, pred_labels)

    # Print intermediate value for debugging
    print("1 - ARI_batch (batch_score):", batch_score)

    # Calculate F1_ARI as per the PDF definition
    denominator = (batch_score + ari_celltype)
    if denominator == 0:
        f1_ari = np.nan
    else:
        f1_ari = 2 * (batch_score * ari_celltype) / denominator

    return ari_celltype, batch_score, f1_ari