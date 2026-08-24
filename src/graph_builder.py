import numpy as np
from .features import get_aa_features, AROMATIC, POSITIVE_CHARGED, NEGATIVE_CHARGED
from .features import HYDROPHOBIC, POLAR_UNCHARGED

def get_residue_type(aa):
    if aa in AROMATIC:
        return 'aromatic'
    elif aa in POSITIVE_CHARGED:
        return 'positive'
    elif aa in NEGATIVE_CHARGED:
        return 'negative'
    elif aa in HYDROPHOBIC:
        return 'hydrophobic'
    elif aa in POLAR_UNCHARGED:
        return 'polar'
    elif aa == 'G':
        return 'glycine'
    elif aa == 'P':
        return 'proline'
    else:
        return 'other'

def get_biochemical_similarity(aa1, aa2):
    if aa1 == aa2:
        return 1.0
    
    type1 = get_residue_type(aa1)
    type2 = get_residue_type(aa2)
    
    if type1 == type2:
        return 0.8
    if (type1 in ['positive', 'negative']) and (type2 in ['positive', 'negative']):
        return 0.6
    if type1 == 'hydrophobic' and type2 == 'hydrophobic':
        return 0.7
    if 'hydrophobic' in [type1, type2] and 'aromatic' in [type1, type2]:
        return 0.6
    if type1 == 'polar' and type2 == 'polar':
        return 0.5
    return 0.2

def build_residue_graph(sequence, k_hop=2, bio_threshold=0.5):
    L = len(sequence)
    
    node_features = [get_aa_features(aa) for aa in sequence]
    
    edge_index = []
    edge_weights = []
    edge_types = []
    edge_distances = []
    
    for i in range(L - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
        edge_weights.append(1.0)
        edge_weights.append(1.0)
        edge_types.append(0)
        edge_types.append(0)
        edge_distances.append(1.0)
        edge_distances.append(1.0)
    
    for d in range(2, k_hop + 1):
        for i in range(L - d):
            edge_index.append([i, i+d])
            edge_index.append([i+d, i])
            weight = 1.0 / d
            edge_weights.append(weight)
            edge_weights.append(weight)
            edge_types.append(1)
            edge_types.append(1)
            edge_distances.append(float(d))
            edge_distances.append(float(d))
    
    for i in range(L):
        for j in range(i + 1, L):
            if j - i <= k_hop:
                continue
            sim = get_biochemical_similarity(sequence[i], sequence[j])
            if sim >= bio_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_weights.append(sim)
                edge_weights.append(sim)
                edge_types.append(2)
                edge_types.append(2)
                edge_distances.append(float(j - i))
                edge_distances.append(float(j - i))
    
    return node_features, edge_index, edge_weights, edge_types, edge_distances
