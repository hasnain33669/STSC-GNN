import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

class MultiScaleGraphPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, x, batch):
        mean_pool = global_mean_pool(x, batch)
        add_pool = global_add_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        return torch.cat([mean_pool, add_pool, max_pool], dim=1)

class NodeLevelBatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
    
    def forward(self, x):
        return self.bn(x)

class TopologicalFeatureExtractor(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=32, out_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU()
        )
    
    def forward(self, edge_index, edge_weight, edge_type, num_nodes):
        if num_nodes == 0 or edge_index.size(1) == 0:
            return torch.zeros(64, device=edge_index.device)
        
        num_edges = edge_index.size(1)
        
        avg_degree = 2 * num_edges / (num_nodes + 1e-8)
        
        num_seq = (edge_type == 0).sum().float()
        num_khop = (edge_type == 1).sum().float()
        num_bio = (edge_type == 2).sum().float()
        
        ratio_seq = num_seq / (num_edges + 1e-8)
        ratio_khop = num_khop / (num_edges + 1e-8)
        ratio_bio = num_bio / (num_edges + 1e-8)
        
        avg_weight = edge_weight.mean()
        
        num_undirected = num_edges / 2.0
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_undirected / (max_possible_edges + 1e-8)
        
        features = torch.cat([
            torch.tensor([num_edges, avg_degree, avg_weight,
                         ratio_seq, ratio_khop, ratio_bio, density],
                        device=edge_index.device)
        ])
        
        return self.proj(features)

class BaseSTSCModel(nn.Module):
    def __init__(self, node_input_dim=30, esm_dim=640, hidden_dim=384,
                 output_dim=3, num_gnn_layers=5, dropout=0.1, variant='base'):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.variant = variant
        
        self.aa_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.gnn = None
        self.graph_pool = MultiScaleGraphPooling(hidden_dim)
        
        self.topology_extractor = TopologicalFeatureExtractor(
            in_dim=7, hidden_dim=32, out_dim=64
        )
        
        self.solvent_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.GELU()
        )
        
        self.solvent_gate = nn.Sequential(
            nn.Linear(hidden_dim + 128, hidden_dim),
            nn.Sigmoid()
        )
        
        self.graph_dim = hidden_dim * 3
        self.seq_to_solvent_attention = nn.MultiheadAttention(
            embed_dim=self.graph_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
            kdim=128,
            vdim=128
        )
        self.solvent_to_seq_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
            kdim=self.graph_dim,
            vdim=self.graph_dim
        )
        
        self.graph_proj = nn.Linear(self.graph_dim, hidden_dim)
        self.solvent_proj = nn.Linear(128, 128)
        
        self.predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim + 128 + 64),
            nn.Linear(hidden_dim + 128 + 64, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x, edge_index, edge_weight, edge_type, edge_distance,
                solvent, batch, return_logits=False):
        aa_features = x[:, :30]
        esm_features = x[:, 30:]
        
        aa_proj = self.aa_proj(aa_features)
        esm_proj = self.esm_proj(esm_features)
        node_features = torch.cat([aa_proj, esm_proj], dim=1)
        
        gnn_out = self.gnn(node_features, edge_index, edge_weight, edge_type, edge_distance)
        
        graph_embedding = self.graph_pool(gnn_out, batch)
        
        batch_size = torch.max(batch).item() + 1
        
        if len(solvent.shape) == 1:
            if solvent.numel() == batch_size * 4:
                solvent = solvent.view(batch_size, 4)
            else:
                solvent = solvent.view(-1, 4)
                if solvent.size(0) == 1 and batch_size > 1:
                    solvent = solvent.expand(batch_size, -1)
        
        topo_features_list = []
        num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
        node_shifts = torch.cat([torch.tensor([0], device=batch.device), 
                                 num_nodes_per_graph.cumsum(dim=0)[:-1]])
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes_i = num_nodes_per_graph[i].item()
            
            if num_nodes_i > 0:
                edge_mask_src = mask[edge_index[0]]
                edge_mask_dst = mask[edge_index[1]]
                edge_mask = edge_mask_src & edge_mask_dst
                
                edge_idx_i = edge_index[:, edge_mask]
                edge_weight_i = edge_weight[edge_mask]
                edge_type_i = edge_type[edge_mask]
                edge_distance_i = edge_distance[edge_mask]
                
                if edge_idx_i.numel() > 0:
                    local_edge_idx_i = edge_idx_i - node_shifts[i]
                else:
                    local_edge_idx_i = edge_idx_i
                
                topo_i = self.topology_extractor(local_edge_idx_i, edge_weight_i, 
                                                  edge_type_i, num_nodes_i)
            else:
                topo_i = torch.zeros(64, device=x.device)
            
            topo_features_list.append(topo_i)
        
        topo_features = torch.stack(topo_features_list)
        
        solvent_embedding = self.solvent_encoder(solvent)
        
        graph_seq = graph_embedding.unsqueeze(1)
        solvent_seq = solvent_embedding.unsqueeze(1)
        
        solvent_attended_seq, _ = self.seq_to_solvent_attention(
            graph_seq, solvent_seq, solvent_seq
        )
        solvent_attended_seq = solvent_attended_seq.squeeze(1)
        
        seq_attended_solvent, _ = self.solvent_to_seq_attention(
            solvent_seq, graph_seq, graph_seq
        )
        seq_attended_solvent = seq_attended_solvent.squeeze(1)
        
        combined_seq = graph_embedding + solvent_attended_seq
        combined_solvent = solvent_embedding + seq_attended_solvent
        
        combined_seq_proj = self.graph_proj(combined_seq)
        combined_solvent_proj = self.solvent_proj(combined_solvent)
        
        gate_input = torch.cat([combined_seq_proj, combined_solvent_proj], dim=1)
        gate = self.solvent_gate(gate_input)
        graph_conditioned = gate * combined_seq_proj
        
        fusion_input = torch.cat([
            graph_conditioned,
            combined_solvent_proj,
            topo_features
        ], dim=1)
        
        logits = self.predictor(fusion_input)
        
        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=1)
