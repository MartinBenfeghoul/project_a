import torch

# Surprise-based
def find_thresholds(surprise, window=100, threshold_param=3.0, min_size=0, fixed_prob=None):

    events_sur = []
    if fixed_prob is None:
        threshold = torch.zeros(len(surprise))
        for t in range(len(surprise)):
            if 0 < t <= window:
                threshold[t] = torch.mean(surprise[:t]) + torch.std(surprise[:t])*threshold_param
            elif t > window:
                threshold[t] = torch.mean(surprise[t-window:t]) + torch.std(surprise[t-window:t])*threshold_param
            else:
                threshold[t] = surprise[t]  # For t == 0
    else:
        threshold = torch.full((len(surprise),),-torch.log(torch.tensor(fixed_prob)))
        threshold[0] = surprise[0]

    for t in range(len(surprise)):
        if surprise[t] >= threshold[t]:
            if len(events_sur) == 0:
                events_sur.append(t)
            elif t - events_sur[-1] >= min_size:
                events_sur.append(t)
    # Append last timestep as a border
    if events_sur[-1] != t:
        events_sur.append(t)

    
    print('The threshold found',len(events_sur),'events including borders.')
    return events_sur, threshold

# K-Sim 
def sequential_segmentation(keys, gamma=1.0, min_event_len=4):
    """Sequential K-Sim Segmentation as per original paper."""
    T, D = keys.shape
    boundaries = [0]
    t0 = 0
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
            hist_tensor = torch.tensor(score_history, device=keys.device, dtype=keys.dtype)
            
            mu = hist_tensor.mean().item()
            sigma = hist_tensor.std().item()
            threshold = mu - gamma * sigma
            if score < threshold:
                is_boundary = True
        
        if is_boundary:
            boundaries.append(t)
            
            # Reset Accumulators for new event
            t0 = t
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
    P_shifted = torch.cat([torch.zeros(1, D, device=keys.device), P[:-1]], dim=0)
    alpha_global = (P_shifted * keys).sum(dim=1)
    while t0 < T:        
        tail_indices = torch.arange(t0 + 1, T, device=keys.device)
        if len(tail_indices) == 0: break
        P_prev = P[tail_indices - 1]
        
        # sum up to before current event started
        P_start = P[t0 - 1] if t0 > 0 else torch.zeros_like(P[0])            
        alpha = alpha_global[tail_indices]
        beta = (P_start * keys[tail_indices]).sum(dim=1) # P[t0-1] . k_t
        
        # Count of items in the mean calculation
        counts = (tail_indices - t0).float()
        
        # The score vector for the entire tail
        v = (alpha - beta) / counts

        # population statistics via cumulative sums
        cum_v = torch.cumsum(v, 0)
        cum_v2 = torch.cumsum(v**2, 0)
        mu_numer = cum_v[:-1]
        sq_numer = cum_v2[:-1]
        ns = torch.arange(1, len(v), device=keys.device).float() # 1, 2, 3...
        mu = mu_numer / ns # mean
        var_pop = (sq_numer / ns) - (mu ** 2) # variance
        
        # Bessel correction for sample variance (matching torch.std)
        # var_sample = var_pop * (N / N-1), where n > 1
        valid_mask = ns > 1
        var_sample = torch.zeros_like(var_pop)
        var_sample[valid_mask] = var_pop[valid_mask] * (ns[valid_mask] / (ns[valid_mask] - 1))
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