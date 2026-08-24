import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseSTSCModel, NodeLevelBatchNorm

class GINWithEdgeFeatures(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=3):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
        self.edge_transform = nn.Linear(edge_channels, in_channels)
        self.edge_type_weights = nn.Parameter(torch.ones(3) / 3)
        self.norm = NodeLevelBatchNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, edge_index, edge_weight, edge_type, edge_distance):
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        type_weights = self.edge_type_weights[edge_type]
        edge_features = torch.stack([
            edge_weight,
            type_weights,
            edge_distance / 10.0
        ], dim=1)
        
        edge_contrib = self.edge_transform(edge_features)
        edge_contrib_aggregated = torch.zeros(num_nodes, self.mlp[0].in_features, device=x.device)
        edge_contrib_aggregated.index_add_(0, edge_index[0], edge_contrib)
        
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float().unsqueeze(1)
        edge_contrib_aggregated = edge_contrib_aggregated / (deg + 1e-8)
        
        x_combined = x + edge_contrib_aggregated
        out = self.mlp(x_combined)
        
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out

class GINModel(BaseSTSCModel):
    def __init__(self, node_input_dim=30, esm_dim=640, hidden_dim=384,
                 output_dim=3, num_gnn_layers=5, dropout=0.1):
        super().__init__(node_input_dim, esm_dim, hidden_dim, output_dim, 
                         num_gnn_layers, dropout, variant='gin')
        
        self.gnn = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = hidden_dim if i > 0 else hidden_dim
            self.gnn.append(GINWithEdgeFeatures(in_channels, hidden_dim))
        
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index, edge_weight, edge_type, edge_distance,
                solvent, batch, return_logits=False):
        aa_features = x[:, :30]
        esm_features = x[:, 30:]
        
        aa_proj = self.aa_proj(aa_features)
        esm_proj = self.esm_proj(esm_features)
        node_features = torch.cat([aa_proj, esm_proj], dim=1)
        
        for conv in self.gnn:
            node_features = conv(node_features, edge_index, edge_weight, 
                                 edge_type, edge_distance)
        
        node_features = self.proj_out(node_features)
        
        return self._forward_from_features(node_features, edge_index, edge_weight,
                                           edge_type, edge_distance, solvent, batch,
                                           return_logits)
    
    def _forward_from_features(self, node_features, edge_index, edge_weight,
                               edge_type, edge_distance, solvent, batch, return_logits):
        graph_embedding = self.graph_pool(node_features, batch)
        
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
                topo_i = torch.zeros(64, device=node_features.device)
            
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
