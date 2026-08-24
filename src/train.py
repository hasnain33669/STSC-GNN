import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import AverageMeter, fix_batch_shapes

def compute_huber_loss(pred, target, delta=1.0):
    diff = pred - target
    abs_diff = torch.abs(diff)
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    return 0.5 * quadratic**2 + delta * linear

def train_epoch(model, train_loader, optimizer, criterion, device,
                plasticity_weight=0.3, huber_delta=1.0):
    model.train()
    running_loss = AverageMeter()
    running_huber_loss = AverageMeter()
    running_plasticity_loss = AverageMeter()
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        batch = fix_batch_shapes(batch)
        
        probs = model(
            batch.x, batch.edge_index, batch.edge_weight,
            batch.edge_type, batch.edge_distance,
            batch.solvent, batch.batch,
            return_logits=False
        )
        
        targets = batch.y
        
        huber_loss = compute_huber_loss(probs, targets, delta=huber_delta).mean()
        
        pred_jsd = F.kl_div(
            F.log_softmax(probs, dim=-1),
            F.softmax(targets, dim=-1),
            reduction='batchmean'
        )
        
        loss = huber_loss + plasticity_weight * pred_jsd
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss.update(loss.item(), targets.size(0))
        running_huber_loss.update(huber_loss.item(), targets.size(0))
        running_plasticity_loss.update(pred_jsd.item(), targets.size(0))
    
    return running_loss.avg, running_huber_loss.avg, running_plasticity_loss.avg

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, device,
                epochs=100, model_name="STSCGT", plasticity_weight=0.3):
    train_losses = []
    val_losses = []
    val_accs = []
    train_huber_losses = []
    train_plasticity_losses = []
    
    best_val_acc = 0.0
    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/best_{model_name}.pt'
    
    print("=" * 100)
    print(f"{'Epoch':<8} {'Train_Loss':<12} {'Huber':<12} {'Plast':<12} "
          f"{'Val_Loss':<12} {'Val_Acc':<12} {'LR':<12}")
    print("=" * 100)
    
    for epoch in range(epochs):
        train_loss, huber_loss, plast_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, plasticity_weight
        )
        
        val_loss, val_mae, val_rmse, val_acc, _, _, _ = evaluate(model, val_loader, device, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        train_huber_losses.append(huber_loss)
        train_plasticity_losses.append(plast_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"{(epoch+1):<8} {train_loss:<12.6f} {huber_loss:<12.6f} {plast_loss:<12.6f} "
              f"{val_loss:<12.6f} {val_acc:<12.6f} {current_lr:<12.2e}")
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.4f})")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"\n✅ Loaded best model from {model_path}")
    
    return model, train_losses, val_losses, val_accs

def evaluate(model, dataloader, device, criterion=None, huber_delta=1.0):
    from .evaluate import evaluate_model
    return evaluate_model(model, dataloader, device, criterion, huber_delta)
