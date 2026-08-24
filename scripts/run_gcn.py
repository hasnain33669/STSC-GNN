import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch_geometric.loader import DataLoader

from src.data_loader import load_data, SequenceTopologyPeptideDataset
from src.features import generate_esm2_embeddings
from src.models import GCNModel
from src.train import train_model, evaluate
from src.utils import set_seed, count_parameters

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dict = load_data('./data')
    if data_dict is None:
        print("❌ Data loading failed!")
        return
    
    train_df, valid_df, test_df, temporal_1_df, temporal_2_df = (
        data_dict['train'], data_dict['valid'], data_dict['test'],
        data_dict['temporal_1'], data_dict['temporal_2']
    )
    
    all_sequences = set()
    for df in [train_df, valid_df, test_df, temporal_1_df, temporal_2_df]:
        all_sequences.update(df['peptide'].values)
    
    cache_path = 'esm2_embeddings_cache.pt'
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        peptide_embeddings = torch.load(cache_path, weights_only=False)
    else:
        peptide_embeddings = generate_esm2_embeddings(list(all_sequences))
        torch.save(peptide_embeddings, cache_path)
    
    train_dataset = SequenceTopologyPeptideDataset(
        root='./peptide_data', df=train_df,
        peptide_embeddings=peptide_embeddings, types='train'
    )
    val_dataset = SequenceTopologyPeptideDataset(
        root='./peptide_data', df=valid_df,
        peptide_embeddings=peptide_embeddings, types='val'
    )
    test_dataset = SequenceTopologyPeptideDataset(
        root='./peptide_data', df=test_df,
        peptide_embeddings=peptide_embeddings, types='test'
    )
    temporal_1_dataset = SequenceTopologyPeptideDataset(
        root='./peptide_data', df=temporal_1_df,
        peptide_embeddings=peptide_embeddings, types='temporal_1'
    )
    temporal_2_dataset = SequenceTopologyPeptideDataset(
        root='./peptide_data', df=temporal_2_df,
        peptide_embeddings=peptide_embeddings, types='temporal_2'
    )
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    temporal_1_loader = DataLoader(temporal_1_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    temporal_2_loader = DataLoader(temporal_2_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Temporal-1: {len(temporal_1_dataset)}")
    print(f"  Temporal-2: {len(temporal_2_dataset)}")
    
    model = GCNModel(
        node_input_dim=30,
        esm_dim=640,
        hidden_dim=384,
        output_dim=3,
        num_gnn_layers=5,
        dropout=0.1
    ).to(device)
    
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )
    criterion = torch.nn.HuberLoss(delta=1.0)
    
    model_name = "STSCGT_GCN"
    model, train_losses, val_losses, val_accs = train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device,
        epochs=100, model_name=model_name, plasticity_weight=0.3
    )
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    test_loss, test_mae, test_rmse, test_acc, _, _, _ = evaluate(
        model, test_loader, device, criterion
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test Accuracy (δ=0.10): {test_acc:.4f}")
    
    print("\n" + "="*60)
    print("TEMPORAL PANELS EVALUATION")
    print("="*60)
    
    temporal_1_loss, temporal_1_mae, temporal_1_rmse, temporal_1_acc, _, _, _ = evaluate(
        model, temporal_1_loader, device, criterion
    )
    print(f"Temporal-1 - Loss: {temporal_1_loss:.4f}, MAE: {temporal_1_mae:.4f}, Acc: {temporal_1_acc:.4f}")
    
    temporal_2_loss, temporal_2_mae, temporal_2_rmse, temporal_2_acc, _, _, _ = evaluate(
        model, temporal_2_loader, device, criterion
    )
    print(f"Temporal-2 - Loss: {temporal_2_loss:.4f}, MAE: {temporal_2_mae:.4f}, Acc: {temporal_2_acc:.4f}")
    
    print("\n✅ GCN training completed!")

if __name__ == "__main__":
    main()
