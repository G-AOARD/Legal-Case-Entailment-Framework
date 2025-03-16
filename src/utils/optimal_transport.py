import ot
import numpy as np

import torch
import torch.nn.functional as F

from rank_bm25 import BM25Okapi

from src.utils import top_k_largest_index
from src.utils.text_utils import word_tokenize


def min_max_scaling(C):
    eps = 1e-10
    nx = ot.backend.get_backend(C)
    C_min = nx.min(C)
    C_max = nx.max(C)
    C = (C - C_min + eps) / (C_max - C_min + eps)
    return C


def compute_embeddings_alignment(
    query_embeddings, doc_embeddings, ot_configs, query_text=None, doc_text=None, query_words=None,
    doc_words=None, query_word_ids=None, doc_word_ids=None, corpus=None, subword_weights=1.
):
    if ot_configs["weights_dist"] == "uniform":
        query_H, doc_H = compute_weights_uniform(
            query_embeddings, doc_embeddings, query_word_ids=query_word_ids, doc_word_ids=doc_word_ids,
            subword_weights=subword_weights
        )
    elif ot_configs["weights_dist"] == "norm":
        query_H, doc_H = compute_weights_norm(
            query_embeddings, doc_embeddings
        )
    else:
        raise ValueError(ot_configs["weights_dist"])

    ot_dist, ot_mat, cost_mat = compute_ot(
        query_embeddings, doc_embeddings, query_H, doc_H, ot_configs,
        cost_distance=ot_configs["cost_distance"], rescale_cost=ot_configs.get("rescale_cost", False),
        distortion_ratio=ot_configs.get("distortion_ratio", 0)
    )
    ot_mat = ot_mat.numpy()
    if ot_configs.get("uniform_mass", False):
        _ot_mat = np.ones_like(ot_mat)
    else:
        _ot_mat = ot_mat
    u_ot_mat = np.zeros_like(ot_mat)
    if ot_configs.get("alignment_threshold", -1) >= 0:
        u_ot_indices = np.argwhere(ot_mat >= ot_configs["alignment_threshold"])
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        u_ot_mat[rows, cols] = _ot_mat[rows, cols]
    if ot_configs.get("alignment_top_k", -1) > 0:
        k = int(min(ot_configs["alignment_top_k"], ot_mat.shape[0] * ot_mat.shape[1]))
        u_ot_indices = top_k_largest_index(ot_mat, k)
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        u_ot_mat[rows, cols] = _ot_mat[rows, cols]
    if ot_configs.get("alignment_top_k_incl", -1) > 0:
        k = int(min(ot_configs["alignment_top_k_incl"], ot_mat.shape[0] * ot_mat.shape[1]))
        u_ot_indices = top_k_largest_index(ot_mat, k)
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        top_k_min_val = min(ot_mat[rows, cols])
        u_ot_indices = np.argwhere(ot_mat >= top_k_min_val)
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        u_ot_mat[rows, cols] = _ot_mat[rows, cols]
    if ot_configs.get("alignment_top_p", -1) > 0:
        top_p = np.percentile(ot_mat.flatten(), ot_configs["alignment_top_p"])
        u_ot_indices = np.argwhere(ot_mat >= top_p)
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        u_ot_mat[rows, cols] = _ot_mat[rows, cols]
    if ot_configs.get("alignment_top_k_per_row", -1) > 0:
        top_k_indices = np.argsort(-ot_mat, axis=1)[:, :ot_configs["alignment_top_k_per_row"]]
        _r = np.repeat(list(range(u_ot_mat.shape[0])), ot_configs["alignment_top_k_per_row"])
        u_ot_indices = np.column_stack((_r, top_k_indices.flatten()))
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        u_ot_mat[rows, cols] = _ot_mat[rows, cols]
    if ot_configs.get("alignment_top_k_cost", -1) > 0:
        top_k_indices = np.argsort(cost_mat, axis=1)[:, :ot_configs["alignment_top_k_cost"]]
        _r = np.repeat(list(range(u_ot_mat.shape[0])), ot_configs["alignment_top_k_cost"])
        u_ot_indices = np.column_stack((_r, top_k_indices.flatten()))
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        u_ot_mat[rows, cols] = _ot_mat[rows, cols]
    if ot_configs.get("final_threshold", -1) > 0:
        u_ot_indices = np.argwhere(u_ot_mat >= ot_configs["final_threshold"])
        rows, cols = u_ot_indices[:, 0].tolist(), u_ot_indices[:, 1].tolist()
        _u_ot_mat = np.zeros_like(ot_mat)
        _u_ot_mat[rows, cols] = u_ot_mat[rows, cols]
        u_ot_mat = _u_ot_mat

    u_ot_dist = float(np.sum(u_ot_mat * cost_mat.numpy()))
    return u_ot_dist, u_ot_mat


def compute_weights_uniform(
    query_embeddings, doc_embeddings, norm="l1", query_word_ids=None, doc_word_ids=None,
    subword_weights=0.5
):
    if query_word_ids is not None:
        query_weights = []
        tmp = [query_word_ids[0]]
        for i in range(1, len(query_word_ids)):
            if query_word_ids[i] == tmp[-1]:
                tmp.append(query_word_ids[i])
            else:
                if len(tmp) > 1:
                    query_weights.extend([subword_weights] * len(tmp))
                else:
                    query_weights.append(1)
                tmp = [query_word_ids[i]]

        if len(tmp) > 1:
            query_weights.extend([subword_weights] * len(tmp))
        else:
            query_weights.append(1)
        query_weights = np.array(query_weights)
    else:
        query_weights = np.ones([query_embeddings.shape[0]])

    if doc_word_ids is not None:
        doc_weights = []
        tmp = [doc_word_ids[0]]
        for i in range(1, len(doc_word_ids)):
            if doc_word_ids[i] == tmp[-1]:
                tmp.append(doc_word_ids[i])
            else:
                if len(tmp) > 1:
                    doc_weights.extend([subword_weights] * len(tmp))
                else:
                    doc_weights.append(1)
                tmp = [doc_word_ids[i]]

        if len(tmp) > 1:
            doc_weights.extend([subword_weights] * len(tmp))
        else:
            doc_weights.append(1)
        doc_weights = np.array(doc_weights)
    else:
        doc_weights = np.ones([doc_embeddings.shape[0]])

    if norm == "l2":
        query_weights /= np.linalg.norm(query_weights)
        doc_weights /= np.linalg.norm(doc_weights)
    return query_weights, doc_weights


def compute_weights_norm(query_embeddings, doc_embeddings):
    query_weights = torch.norm(query_embeddings, dim=1).cpu().numpy()
    doc_weights = torch.norm(doc_embeddings, dim=1).cpu().numpy()
    return query_weights, doc_weights


def apply_distortion(sim_matrix, ratio=0.1):
    shape = sim_matrix.shape
    if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
        return sim_matrix

    pos_x = torch.tensor([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])],
                         device=sim_matrix.device)
    pos_y = torch.tensor([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])],
                         device=sim_matrix.device)
    distortion_mask = 1.0 - ((pos_x - pos_y.T) ** 2) * ratio
    sim_matrix = torch.mul(sim_matrix, distortion_mask.to(sim_matrix.device))
    return sim_matrix


def compute_cost_mat(
    X_1, X_2, cost_distance, rescale_cost=False, distortion_ratio=0.
):
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()

    if cost_distance == "l2":
        X_1 = X_1.view(n_1, 1, -1)
        X_2 = X_2.view(1, n_2, -1)
        squared_dist = (X_1 - X_2) ** 2
        C = torch.sum(squared_dist, dim=2)
        C = 1 - min_max_scaling(C)
    elif cost_distance == "cosine":
        C = (torch.matmul(F.normalize(X_1.float()), F.normalize(X_2.float()).t()) + 1.0) / 2
    elif cost_distance == "dot":
        C = X_1 @ X_2.to(dtype=X_1.dtype).transpose(0, 1)
    else:
        assert False, cost_distance

    if distortion_ratio > 0:
        C = apply_distortion(C, distortion_ratio)

    if rescale_cost:
        C = 1 - min_max_scaling(C)
    else:
        C = -C
    return C


def compute_ot(
    X_1, X_2, H_1, H_2, ot_configs, cost_distance='l2', rescale_cost=False, distortion_ratio=0.,
):
    cost_mat = compute_cost_mat(
        X_1, X_2, cost_distance, rescale_cost=rescale_cost, distortion_ratio=distortion_ratio
    )

    cost_mat_detach = cost_mat.detach().cpu().numpy()
    if isinstance(ot_configs["reg_m"], int):
        reg_m = ot_configs["reg_m"]
    else:
        if ot_configs["reg_m"] == "r1":
            reg_m = float("inf")
        elif ot_configs["reg_m"] == "r2":
            reg_m = [1, float("inf")]
        else:
            reg_m = [float("inf"), 1]

    
    if ot_configs["opt_method"] == "sinkhorn_knopp_unbalanced":
        ot_mat = ot.unbalanced.sinkhorn_knopp_unbalanced(
            a=H_1, b=H_2, M=cost_mat_detach,
            reg=ot_configs["reg"],
            reg_m=reg_m,
            reg_type=ot_configs.get("reg_type", "entropy"),
            warmstart=ot_configs.get("warmstart", None),
            numItermax=ot_configs.get("max_iter", 1000),
        )
    elif ot_configs["opt_method"] == "sinkhorn_stabilized_unbalanced":
        ot_mat = ot.unbalanced.sinkhorn_stabilized_unbalanced(
            a=H_1, b=H_2, M=cost_mat_detach,
            reg=ot_configs["reg"],
            reg_m=reg_m,
            reg_type=ot_configs.get("reg_type", "entropy"),
            warmstart=ot_configs.get("warmstart", None),
            tau=ot_configs.get("tau", 100000.0),
            numItermax=ot_configs.get("max_iter", 1000),
        )
    else:
        raise ValueError(ot_configs["opt_method"])

    ot_mat_attached = torch.tensor(
        ot_mat, device=cost_mat.device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached, cost_mat
