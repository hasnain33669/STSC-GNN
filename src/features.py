import torch
import esm
from tqdm import tqdm
import numpy as np

VOCAB_PROTEIN = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 4,
    "G": 5, "H": 6, "I": 7, "K": 8, "L": 9,
    "M": 10, "N": 11, "P": 12, "Q": 13, "R": 14,
    "S": 15, "T": 16, "V": 17, "W": 18, "Y": 19
}

HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'E': -3.5, 'Q': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'K': -3.9, 'L': 3.8, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

POSITIVE_CHARGED = {'K', 'R', 'H'}
NEGATIVE_CHARGED = {'D', 'E'}
HYDROPHOBIC = {'A', 'I', 'L', 'M', 'F', 'V', 'W', 'C'}
POLAR_UNCHARGED = {'N', 'Q', 'S', 'T', 'Y'}
AROMATIC = {'F', 'Y', 'W'}

def get_aa_features(aa):
    one_hot = [0.0] * 20
    if aa in VOCAB_PROTEIN:
        one_hot[VOCAB_PROTEIN[aa]] = 1.0
    else:
        one_hot[VOCAB_PROTEIN['G']] = 1.0
    
    phys = [0.0] * 10
    phys[0] = HYDROPHOBICITY.get(aa, 0.0)
    phys[1] = 1.0 if aa in POSITIVE_CHARGED else 0.0
    phys[2] = 1.0 if aa in NEGATIVE_CHARGED else 0.0
    phys[3] = 1.0 if aa in HYDROPHOBIC else 0.0
    phys[4] = 1.0 if aa in POLAR_UNCHARGED else 0.0
    phys[5] = 1.0 if aa in AROMATIC else 0.0
    phys[6] = 1.0 if aa == 'G' else 0.0
    phys[7] = 1.0 if aa == 'P' else 0.0
    phys[8] = 1.0 if aa in {'A', 'V', 'I', 'L', 'M'} else 0.0
    phys[9] = 1.0 if aa in {'K', 'R'} else 0.0
    
    return np.array(one_hot + phys, dtype=np.float32)

def get_aa_embedding_size():
    return 20 + 10

def generate_esm2_embeddings(sequences, batch_size=1):
    print("Loading ESM-2 model (150M parameters)...")
    
    model, alphabet = esm.pretrained.load_model_and_alphabet('esm2_t30_150M_UR50D')
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    embeddings_dict = {}
    sequences_list = list(sequences)
    
    print(f"Generating embeddings for {len(sequences_list)} unique sequences...")
    
    for i in tqdm(range(0, len(sequences_list), batch_size), desc="ESM-2"):
        batch_seqs = sequences_list[i:i+batch_size]
        batch_data = [(f"peptide_{j}", seq) for j, seq in enumerate(batch_seqs)]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[30])
            token_representations = results["representations"][30]
            
            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                residue_embeddings = token_representations[j, 1:seq_len+1, :]
                embeddings_dict[seq] = residue_embeddings.cpu()
    
    print(f"✅ Generated embeddings for {len(embeddings_dict)} sequences")
    return embeddings_dict
