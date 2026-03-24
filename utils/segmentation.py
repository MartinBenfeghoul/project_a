import torch


def _infllm_init(features: torch.Tensor, n_clusters: int) -> torch.Tensor:
    init_idx = torch.arange(
        n_clusters, device=features.device, dtype=torch.long
    )
    init_idx = torch.div(
        init_idx * features.size(1), n_clusters, rounding_mode="floor"
    )
    return features[:, init_idx, :].clone()


def _random_init(features: torch.Tensor, n_clusters: int) -> torch.Tensor:
    batch_size, _, dim = features.shape
    indices = torch.stack(
        [
            torch.randperm(features.size(1), device=features.device)[
                :n_clusters
            ]
            for _ in range(batch_size)
        ]
    )
    gather_idx = indices.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(features, 1, gather_idx).clone()


def _kmeans_pp_init(features: torch.Tensor, n_clusters: int) -> torch.Tensor:
    batch_size, seq_len, _ = features.shape
    first_idx = torch.randint(
        seq_len, (batch_size,), device=features.device
    )
    centroids = [
        features[torch.arange(batch_size, device=features.device), first_idx]
    ]

    for _ in range(1, n_clusters):
        current_centroids = torch.stack(centroids, dim=1)
        min_distances = torch.cdist(
            features, current_centroids
        ).min(dim=-1).values
        probs = min_distances.square()
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        next_idx = torch.multinomial(probs, 1).squeeze(-1)
        centroids.append(
            features[
                torch.arange(batch_size, device=features.device), next_idx
            ]
        )

    return torch.stack(centroids, dim=1)


KMEANS_INIT_METHODS = {
    "infllm": _infllm_init,
    "random": _random_init,
    "kmeans++": _kmeans_pp_init,
}


def kmeans_cluster_sequences(
    features: torch.Tensor,
    n_clusters: int,
    n_iter: int = 8,
    kmeans_init: str = "infllm",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Cluster per-sequence token features with a small batched KMeans loop."""
    if features.dim() != 3:
        raise ValueError(
            "kmeans_cluster_sequences expects features with shape "
            "[batch, seq_len, feature_dim]."
        )

    batch_size, seq_len, _ = features.shape
    if seq_len == 0:
        return torch.empty(
            batch_size, 0, dtype=torch.long, device=features.device
        )

    effective_clusters = max(1, min(int(n_clusters), seq_len))
    if effective_clusters == 1:
        return torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=features.device
        )

    work_features = features.to(dtype=dtype)
    try:
        init_fn = KMEANS_INIT_METHODS[kmeans_init]
    except KeyError as exc:
        raise ValueError(
            f"Unknown kmeans init method: {kmeans_init}. "
            f"Available methods: {sorted(KMEANS_INIT_METHODS)}"
        ) from exc
    centroids = init_fn(work_features, effective_clusters)

    prev_labels = None
    ones = torch.ones(
        batch_size, seq_len, dtype=work_features.dtype, device=features.device
    )
    for _ in range(n_iter):
        distances = torch.cdist(work_features, centroids)
        labels = distances.argmin(dim=-1)
        if prev_labels is not None and torch.equal(labels, prev_labels):
            break

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(
            batch_size,
            effective_clusters,
            dtype=work_features.dtype,
            device=features.device,
        )
        scatter_idx = labels.unsqueeze(-1).expand(
            -1, -1, work_features.size(-1)
        )
        new_centroids.scatter_add_(1, scatter_idx, work_features)
        counts.scatter_add_(1, labels, ones)

        empty_mask = counts == 0
        new_centroids = new_centroids / counts.clamp_min(1).unsqueeze(-1)
        new_centroids[empty_mask] = centroids[empty_mask]

        centroids = new_centroids
        prev_labels = labels

    return labels


def build_cluster_segment_ranges(
    cluster_assignments: torch.Tensor,
    n_clusters: int | None = None,
) -> list[list[tuple[int, int]]]:
    """Build contiguous segment ranges for cluster-grouped sequences."""
    if cluster_assignments.dim() != 2:
        raise ValueError(
            "build_cluster_segment_ranges expects assignments with shape "
            "[batch, seq_len]."
        )

    if cluster_assignments.size(-1) == 0:
        return [[] for _ in range(cluster_assignments.size(0))]

    if n_clusters is None:
        n_clusters = int(cluster_assignments.max().item()) + 1

    segment_ranges = []
    for batch_assignments in cluster_assignments:
        batch_ranges = []
        counts = torch.bincount(batch_assignments, minlength=n_clusters)
        start_idx = 0
        for count in counts.tolist():
            end_idx = start_idx + count
            if count > 0:
                batch_ranges.append((start_idx, end_idx))
            start_idx = end_idx
        segment_ranges.append(batch_ranges)
    return segment_ranges


def group_keys_by_cluster(
    keys: torch.Tensor,
    cluster_assignments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder keys so tokens assigned to the same cluster are contiguous."""
    if keys.dim() != 4:
        raise ValueError(
            "group_keys_by_cluster expects keys with shape [batch, heads, seq, dim]."
        )
    if cluster_assignments.shape != keys.shape[:1] + keys.shape[2:3]:
        raise ValueError(
            "cluster_assignments must have shape [batch, seq_len] matching keys."
        )

    permutation = torch.argsort(cluster_assignments, dim=-1)
    inverse_permutation = torch.empty_like(permutation)
    original_positions = (
        torch.arange(
            permutation.size(-1),
            device=permutation.device,
            dtype=permutation.dtype,
        )
        .unsqueeze(0)
        .expand_as(permutation)
    )
    inverse_permutation.scatter_(1, permutation, original_positions)

    gather_idx = permutation[:, None, :, None].expand_as(keys)
    grouped_keys = torch.gather(keys, dim=2, index=gather_idx)
    return grouped_keys, permutation, inverse_permutation


def group_sequences_by_cluster(
    sequences: torch.Tensor,
    cluster_assignments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reorder sequence tensors so tokens in the same cluster are contiguous."""
    if sequences.dim() != 3:
        raise ValueError(
            "group_sequences_by_cluster expects tensors with shape "
            "[batch, seq_len, dim]."
        )
    if cluster_assignments.shape != sequences.shape[:2]:
        raise ValueError(
            "cluster_assignments must have shape [batch, seq_len] matching "
            "sequences."
        )

    permutation = torch.argsort(cluster_assignments, dim=-1)
    inverse_permutation = torch.empty_like(permutation)
    original_positions = (
        torch.arange(
            permutation.size(-1),
            device=permutation.device,
            dtype=permutation.dtype,
        )
        .unsqueeze(0)
        .expand_as(permutation)
    )
    inverse_permutation.scatter_(1, permutation, original_positions)

    gather_idx = permutation[:, :, None].expand_as(sequences)
    grouped_sequences = torch.gather(sequences, dim=1, index=gather_idx)
    return grouped_sequences, permutation, inverse_permutation


def restore_grouped_keys_order(
    grouped_keys: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> torch.Tensor:
    """Undo `group_keys_by_cluster` with a stored inverse permutation."""
    if grouped_keys.dim() != 4:
        raise ValueError(
            "restore_grouped_keys_order expects keys with shape "
            "[batch, heads, seq, dim]."
        )
    if (
        inverse_permutation.shape
        != grouped_keys.shape[:1] + grouped_keys.shape[2:3]
    ):
        raise ValueError(
            "inverse_permutation must have shape [batch, seq_len] matching keys."
        )

    gather_idx = inverse_permutation[:, None, :, None].expand_as(grouped_keys)
    return torch.gather(grouped_keys, dim=2, index=gather_idx)


def restore_grouped_sequences_order(
    grouped_sequences: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> torch.Tensor:
    """Undo `group_sequences_by_cluster` with a stored inverse permutation."""
    if grouped_sequences.dim() != 3:
        raise ValueError(
            "restore_grouped_sequences_order expects tensors with shape "
            "[batch, seq_len, dim]."
        )
    if inverse_permutation.shape != grouped_sequences.shape[:2]:
        raise ValueError(
            "inverse_permutation must have shape [batch, seq_len] matching "
            "sequences."
        )

    gather_idx = inverse_permutation[:, :, None].expand_as(grouped_sequences)
    return torch.gather(grouped_sequences, dim=1, index=gather_idx)


# Surprise-based
def find_thresholds(
    surprise, window=100, threshold_param=3.0, min_size=0, fixed_prob=None
):
    # TODO: this code sucks.. (I wrote it) it's unclear and messy - rewrite!
    events_sur = []
    if fixed_prob is None:
        threshold = torch.zeros(len(surprise))
        for t in range(len(surprise)):
            if 0 < t <= window:
                threshold[t] = (
                    torch.mean(surprise[:t])
                    + torch.std(surprise[:t]) * threshold_param
                )
            elif t > window:
                threshold[t] = (
                    torch.mean(surprise[t - window : t])
                    + torch.std(surprise[t - window : t]) * threshold_param
                )
            else:
                threshold[t] = surprise[t]  # For t == 0
    else:
        threshold = torch.full(
            (len(surprise),), -torch.log(torch.tensor(fixed_prob))
        )
        threshold[0] = surprise[0]

    for t in range(len(surprise)):
        if surprise[t] >= threshold[t]:
            if len(events_sur) == 0:
                events_sur.append(t)
            elif t - events_sur[-1] >= min_size:
                events_sur.append(t)
    # Append last timestep as a border - this one should be inclusive
    if t - events_sur[-1] <= min_size:
        events_sur[-1] = t + 1
    else:
        events_sur.append(t + 1)

    print("The threshold found", len(events_sur), "events including borders.")
    return events_sur, threshold


# K-Sim
def sequential_segmentation(keys, gamma=1.0, min_event_len=4):
    """Sequential K-Sim Segmentation as per original paper."""
    T, D = keys.shape
    boundaries = [0]
    event_key_sum = keys[0].clone()
    event_len = 1
    score_history = []

    for t in range(1, T):
        current_key = keys[t]
        mean_key = event_key_sum / event_len

        # compute similarity score (dot product)
        score = torch.dot(mean_key, current_key).item()

        # check for boundary condition
        is_boundary = False
        if len(score_history) >= min_event_len:
            hist_tensor = torch.tensor(
                score_history, device=keys.device, dtype=keys.dtype
            )

            mu = hist_tensor.mean().item()
            sigma = hist_tensor.std().item()
            threshold = mu - gamma * sigma
            if score < threshold:
                is_boundary = True

        if is_boundary:
            boundaries.append(t)

            # Reset Accumulators for new event
            event_key_sum = current_key.clone()
            event_len = 1
            score_history = []

        else:
            event_key_sum += current_key
            event_len += 1
            score_history.append(score)

    return boundaries


def parallel_segmentation(keys, gamma=1.0, min_event_len=4):
    """Parallel K-Sim Segmentation using vectorized operations."""
    T, D = keys.shape
    boundaries = [0]
    t0 = 0

    P = torch.cumsum(keys, dim=0)
    P_shifted = torch.cat(
        [torch.zeros(1, D, device=keys.device), P[:-1]], dim=0
    )
    alpha_global = (P_shifted * keys).sum(dim=1)
    while t0 < T:
        tail_indices = torch.arange(t0 + 1, T, device=keys.device)
        if len(tail_indices) == 0:
            break

        # sum up to before current event started
        P_start = P[t0 - 1] if t0 > 0 else torch.zeros_like(P[0])
        alpha = alpha_global[tail_indices]
        beta = (P_start * keys[tail_indices]).sum(dim=1)  # P[t0-1] . k_t

        # Count of items in the mean calculation
        counts = (tail_indices - t0).float()

        # The score vector for the entire tail
        v = (alpha - beta) / counts

        # population statistics via cumulative sums
        cum_v = torch.cumsum(v, 0)
        cum_v2 = torch.cumsum(v**2, 0)
        mu_numer = cum_v[:-1]
        sq_numer = cum_v2[:-1]
        ns = torch.arange(1, len(v), device=keys.device).float()  # 1, 2, 3...
        mu = mu_numer / ns  # mean
        var_pop = (sq_numer / ns) - (mu**2)  # variance

        # Bessel correction for sample variance (matching torch.std)
        # var_sample = var_pop * (N / N-1), where n > 1
        valid_mask = ns > 1
        var_sample = torch.zeros_like(var_pop)
        var_sample[valid_mask] = var_pop[valid_mask] * (
            ns[valid_mask] / (ns[valid_mask] - 1)
        )
        std = torch.sqrt(torch.clamp(var_sample, min=1e-6))

        # find violations and mask out indices where N < min_len
        thresholds = mu - gamma * std
        violations = v[1:] < thresholds
        mask_min_len = ns >= min_event_len
        violations = violations & mask_min_len

        idx = torch.nonzero(violations, as_tuple=True)[0]

        if len(idx) > 0:
            first_violation_idx = idx[0].item()
            next_t = tail_indices[first_violation_idx + 1].item()
            boundaries.append(next_t)
            t0 = next_t
        else:
            # No boundary found in the remainder of the sequence
            break

    return boundaries
