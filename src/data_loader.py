import os
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from .features import get_aa_features
from .graph_builder import build_residue_graph

def load_data(data_dir="./data"):
    files = {
        'train': os.path.join(data_dir, 'train.csv'),
        'valid': os.path.join(data_dir, 'valid.csv'),
        'test': os.path.join(data_dir, 'test.csv'),
        'temporal_1': os.path.join(data_dir, 'temporal-1.csv'),
        'temporal_2': os.path.join(data_dir, 'temporal-2.csv'),
    }
    
    dataframes = {}
    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            dataframes[name] = df
            print(f"Loaded {name}: {len(df)} samples")
        else:
            print(f"Warning: {path} not found")
            return None
    
    reshaped = {}
    for name, df in dataframes.items():
        reshaped[name] = reshape_to_long(df, name)
    
    return reshaped

def reshape_to_long(df, dataset_name):
    seq_col = 'Sequence' if 'Sequence' in df.columns else 'sequence'
    
    avg_alpha_cols = [col for col in df.columns if 'Alpha_Average' in col]
    solvents = [col.replace('_Alpha_Average', '').replace('_Average', '') 
                for col in avg_alpha_cols]
    
    long_rows = []
    for idx, row in df.iterrows():
        sequence = row[seq_col]
        for solvent in solvents:
            alpha_col = f"{solvent}_Alpha_Average"
            beta_col = f"{solvent}_Beta_Average"
            unstruct_col = f"{solvent}_Unstruct_Average"
            
            if all(col in df.columns for col in [alpha_col, beta_col, unstruct_col]):
                alpha_val = row[alpha_col]
                beta_val = row[beta_col]
                unstruct_val = row[unstruct_col]
                
                if pd.notna(alpha_val) and pd.notna(beta_val) and pd.notna(unstruct_val):
                    long_rows.append({
                        'peptide': sequence,
                        'solvent': solvent,
                        'f_alpha': alpha_val,
                        'f_beta': beta_val,
                        'f_unstructured': unstruct_val
                    })
    
    return pd.DataFrame(long_rows)

class SequenceTopologyPeptideDataset(InMemoryDataset):
    def __init__(self, root, df, peptide_embeddings=None, types='train',
                 transform=None, pre_transform=None):
        self.df = df
        self.types = types
        self.peptide_embeddings = peptide_embeddings
        self.k_hop = 2
        self.bio_threshold = 0.5
        
        super().__init__(root, transform, pre_transform)
        
        type_map = {'train': 0, 'val': 1, 'test': 2, 'temporal_1': 3, 'temporal_2': 4}
        idx = type_map.get(types, 0)
        
        if os.path.exists(self.processed_paths[idx]):
            self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[idx], weights_only=False)
    
    @property
    def raw_file_names(self):
        return ['train.csv', 'valid.csv', 'test.csv', 'temporal-1.csv', 'temporal-2.csv']
    
    @property
    def processed_file_names(self):
        return ['processed_train.pt', 'processed_val.pt', 'processed_test.pt',
                'processed_temporal_1.pt', 'processed_temporal_2.pt']
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        print(f"Processing {self.types} data ({len(self.df)} rows)...")
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Building {self.types}"):
            try:
                sequence = str(row['peptide'])
                L = len(sequence)
                
                node_features, edge_index, edge_weights, edge_types, edge_distances = build_residue_graph(
                    sequence, self.k_hop, self.bio_threshold
                )
                
                if sequence in self.peptide_embeddings:
                    residue_esm = self.peptide_embeddings[sequence]
                    if residue_esm.size(0) < L:
                        pad_size = L - residue_esm.size(0)
                        residue_esm = torch.cat([residue_esm, torch.zeros(pad_size, 640)], dim=0)
                    elif residue_esm.size(0) > L:
                        residue_esm = residue_esm[:L]
                else:
                    residue_esm = torch.zeros(L, 640)
                
                aa_features = torch.FloatTensor(node_features)
                x = torch.cat([aa_features, residue_esm], dim=1)
                
                solvent = str(row['solvent'])
                solvent_tensor = torch.FloatTensor(self.get_solvent_descriptors(solvent))
                
                targets = torch.tensor([
                    float(row['f_alpha']),
                    float(row['f_beta']),
                    float(row['f_unstructured'])
                ], dtype=torch.float32)
                
                data = Data(
                    x=x,
                    edge_index=torch.LongTensor(edge_index).t().contiguous(),
                    edge_weight=torch.FloatTensor(edge_weights),
                    edge_type=torch.LongTensor(edge_types),
                    edge_distance=torch.FloatTensor(edge_distances),
                    y=targets,
                    solvent=solvent_tensor,
                    sequence=sequence,
                    solvent_name=solvent,
                    num_nodes=L,
                    seq_len=L
                )
                
                data_list.append(data)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        type_map = {'train': 0, 'val': 1, 'test': 2, 'temporal_1': 3, 'temporal_2': 4}
        idx = type_map.get(self.types, 0)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[idx])
        print(f"✅ Processed {len(data_list)} samples for {self.types}")
    
    def get_solvent_descriptors(self, solvent):
        solvent_properties = {
            'water': [78.4, 1.17, 0.47, 1.09],
            'MEOH_water': [52.1, 1.02, 0.54, 0.82],
            'SDS': [32.0, 0.40, 0.70, 0.70],
            'TFE_water': [26.7, 1.25, 0.12, 0.62]
        }
        
        solvent_clean = solvent.replace('_Average', '').strip()
        
        if 'water' in solvent_clean.lower() and 'meoh' not in solvent_clean.lower() and 'tfe' not in solvent_clean.lower():
            return solvent_properties['water']
        elif 'meoh' in solvent_clean.lower():
            return solvent_properties['MEOH_water']
        elif 'tfe' in solvent_clean.lower():
            return solvent_properties['TFE_water']
        elif 'sds' in solvent_clean.lower():
            return solvent_properties['SDS']
        else:
            return solvent_properties['water']
