import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KDTree
from statsmodels.robust import mad
from sklearn.neighbors import BallTree

def cca_seurat(X, Y, n_components=50, normalization=True):
    if normalization:
        X = preprocessing.scale(X, axis=0)
        Y = preprocessing.scale(Y, axis=0)
    X = preprocessing.scale(X, axis=1)
    Y = preprocessing.scale(Y, axis=1)
    mat = X @ Y.T
    k = n_components
    u, sig, v = np.linalg.svd(mat, full_matrices=False)
    sigma = np.diag(sig)
    W1 = np.dot(u[:, :k], np.sqrt(sigma[:k, :k]))
    W2 = np.dot(v.T[:, :k], np.sqrt(sigma[:k, :k]))
    return W1, W2

def pca_reduction(X, Y, n_components=50, normalization=True):
    mat = np.vstack([X, Y])
    L1 = len(X)
    if normalization:
        mat = preprocessing.scale(mat)
    model = PCA(n_components=n_components)
    pca_fit = model.fit_transform(mat)
    return pca_fit[0:L1, :], pca_fit[L1:, :]

def kpca_reduction(X, Y, n_components=20, kernel='rbf', normalization=True):
    mat = np.vstack([X, Y])
    L1 = len(X)
    if normalization:
        mat = preprocessing.scale(mat)
    kpca = KernelPCA(n_components=n_components, kernel=kernel)
    kpca_fit = kpca.fit_transform(mat)
    return kpca_fit[0:L1, :], kpca_fit[L1:, :]


from sklearn.neighbors import KDTree
import numpy as np
from statsmodels.robust import mad


def build_weighted_graph(X, k=10, metric='euclidean', sigma=None, sigma_method='median'):
    n_cells = X.shape[0]
    k = min(k, n_cells - 1)  
    if n_cells <= 1:
        raise ValueError("X must contain at least 2 cells.")

    tree = BallTree(X, metric=metric)
    distances, indices = tree.query(X, k=k + 1, return_distance=True)  
    neighbor_dist = distances[:, 1:]  
    neighbor_idxs = indices[:, 1:] 

    mnn_mat = np.zeros((n_cells, n_cells), dtype=bool)
    for i in range(n_cells):
        mnn_mat[i, neighbor_idxs[i]] = True  
    mnn_mat = np.logical_and(mnn_mat, mnn_mat.T) 

    if sigma is None:
        if sigma_method == 'median':
            sigma = np.median(neighbor_dist, axis=1)
        elif sigma_method == 'mad':
            sigma = mad(neighbor_dist, axis=1)
        sigma[sigma <= 0] = 1e-6  

    adjacency_list = [{} for _ in range(n_cells)]
    weight_threshold = 0
    for i in range(n_cells):
        mutual_neighbors = neighbor_idxs[i][mnn_mat[i, neighbor_idxs[i]]]
        mutual_dists = neighbor_dist[i][mnn_mat[i, neighbor_idxs[i]]]
        if len(mutual_neighbors) > 0:
            weights = np.exp(- (mutual_dists ** 2) / (sigma[i] ** 2))  # 高斯核权重
            neighbor_dict = {idx: w for idx, w in zip(mutual_neighbors, weights) if w > weight_threshold}
            adjacency_list[i] = neighbor_dict

    return adjacency_list




def acquire_mnn_pairs(X, Y, k):
    X = preprocessing.normalize(X, axis=1)
    Y = preprocessing.normalize(Y, axis=1)

    t1 = BallTree(X, metric='euclidean')
    t2 = BallTree(Y, metric='euclidean')

    _, idx_xy = t1.query(Y, k=k)
    _, idx_yx = t2.query(X, k=k)

    pairs = set()
    for x_i in range(len(X)):
        y_nbrs = idx_yx[x_i]
        for y_j in y_nbrs:
            if x_i in idx_xy[y_j]:
                pairs.add((x_i, y_j))
    return list(pairs)


def weighted_random_walk_pairs(G_X, G_Y, init_pairs, n_steps, max_pairs=None):
    random_state = np.random.RandomState(999)
    pairs_plus = set() 
    for (x, y) in init_pairs:
        current = (x, y)
        for _ in range(n_steps):
            pairs_plus.add(current)  
            if max_pairs and len(pairs_plus) >= max_pairs:
                break

            x_nbr_dict = G_X[current[0]] if current[0] < len(G_X) else {}
            y_nbr_dict = G_Y[current[1]] if current[1] < len(G_Y) else {}
            if not x_nbr_dict or not y_nbr_dict:
                break

            x_next = weighted_choice(x_nbr_dict, random_state)
            y_next = weighted_choice(y_nbr_dict, random_state)
            current = (x_next, y_next)
        if max_pairs and len(pairs_plus) >= max_pairs:
            break
    print(len(pairs_plus))
    return pairs_plus  

def weighted_choice(weight_dict, random_state):
    """加权随机选择"""
    keys = list(weight_dict.keys())
    weights = list(weight_dict.values())
    total_w = sum(weights)
    if total_w <= 1e-12:
        return random_state.choice(keys)  
    return random_state.choice(keys, p=np.array(weights) / total_w)

def filter_new_pairs(X, Y, pairs, k_filter=10, metric='euclidean', mode='mutual'):
    if not pairs:
        return []
    
    n_x, n_y = X.shape[0], Y.shape[0]
    nbrs_y = NearestNeighbors(n_neighbors=min(k_filter, n_y), metric=metric).fit(Y)
    _, idx_xy = nbrs_y.kneighbors(X)
    nbrs_x = NearestNeighbors(n_neighbors=min(k_filter, n_x), metric=metric).fit(X)
    _, idx_yx = nbrs_x.kneighbors(Y)
    
    filtered = []
    for (xi, yi) in pairs:
        if xi >= n_x or yi >= n_y:
            continue
        x_to_y = yi in idx_xy[xi]
        y_to_x = xi in idx_yx[yi]
        if mode == 'mutual' and x_to_y and y_to_x:
            filtered.append((xi, yi))
        elif mode == 'one-way' and (x_to_y or y_to_x):
            filtered.append((xi, yi))
    return filtered

def modified_rwMNN(X, Y, k_intra=None, k_cross=None, sigma=None, walk_steps=50,
                   filtering=False, k_filter=10, metric='euclidean',
                   reduction=None, norm=True, max_pairs=None):
    subsample = 3000
    if subsample:  
        len_X = len(X)  
        len_Y = len(Y)
        thre = max(len_X, len_Y)
        if thre > subsample:
            tmp = np.arange(len_X)  # [0,...,len_ref-1]
            np.random.shuffle(tmp)
            tmp2 = np.arange(len_Y)
            np.random.shuffle(tmp2)

            X = X[tmp][:subsample] 
            Y = Y[tmp2][:subsample]  

    n_x, n_y = X.shape[0], Y.shape[0]
    if n_x == 0 or n_y == 0:
        return [], []
    
    if k_intra is None:
        k_intra = max(int(min(n_x, n_y) / 100), 5)
    if k_cross is None:
        k_cross = max(int(k_intra / 2), 3)
    print("k_intra", k_intra)
    print("k_cross", k_cross)

    if reduction == 'precomputed':
        x_reduced, y_reduced = X, Y
    elif reduction == 'cca':
        x_reduced, y_reduced = cca_seurat(X, Y, normalization=norm)
    elif reduction == 'pca':
        x_reduced, y_reduced = pca_reduction(X, Y, normalization=norm)
    elif reduction == 'kpca':
        x_reduced, y_reduced = kpca_reduction(X, Y, normalization=norm)
    else:
        x_reduced, y_reduced = X, Y
    
    G_X = build_weighted_graph(x_reduced, k=k_intra, metric=metric, sigma=sigma)
    G_Y = build_weighted_graph(y_reduced, k=k_intra, metric=metric, sigma=sigma)
    
    init_pairs = acquire_mnn_pairs(x_reduced, y_reduced, k=k_cross)
    ref_init_indices = [p[0] for p in init_pairs]  
    query_init_indices = [p[1] for p in init_pairs]  
    
    pairs_plus = weighted_random_walk_pairs(G_X, G_Y, init_pairs, n_steps=walk_steps, 
                                            max_pairs=max_pairs)
    
    if filtering:
        pairs_plus = filter_new_pairs(x_reduced, y_reduced, pairs_plus, k_filter=k_filter, 
                                      metric=metric, mode='one-way')

    ref_index = [p[0] for p in pairs_plus]
    query_index = [p[1] for p in pairs_plus]
    ref_data = X[ref_index, :]
    query_data = Y[query_index, :]
    return ref_data, query_data, ref_index, query_index, (ref_init_indices, query_init_indices)
