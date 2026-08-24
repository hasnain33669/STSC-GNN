import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .utils import AverageMeter, fix_batch_shapes

def compute_accuracy_at_tolerance(pred, target, tolerance=0.10):
    errors = torch.abs(pred - target)
    correct = (errors <= tolerance).float()
    return correct.mean().item()

def evaluate_model(model, dataloader, device, criterion=None, huber_delta=1.0):
    model.eval()
    running_loss = AverageMeter()
    all_preds = []
    all_targets = []
    all_errors = []
    all_accuracy = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            batch = fix_batch_shapes(batch)
            
            probs = model(
                batch.x, batch.edge_index, batch.edge_weight,
                batch.edge_type, batch.edge_distance,
                batch.solvent, batch.batch,
                return_logits=False
            )
            
            targets = batch.y
            
            if criterion:
                loss = criterion(probs, targets)
                running_loss.update(loss.item(), targets.size(0))
            
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            for i in range(probs.size(0)):
                acc = compute_accuracy_at_tolerance(probs[i], targets[i], tolerance=0.10)
                all_accuracy.append(acc)
            
            error = torch.abs(probs - targets)
            all_errors.append(error.cpu().numpy())
    
    if all_preds:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        all_errors = np.vstack(all_errors)
    else:
        all_preds = np.array([])
        all_targets = np.array([])
        all_errors = np.array([])
    
    mae = np.mean(all_errors) if len(all_errors) > 0 else 0
    rmse = np.sqrt(np.mean(all_errors ** 2)) if len(all_errors) > 0 else 0
    acc_mean = np.mean(all_accuracy) if all_accuracy else 0
    
    if criterion:
        return running_loss.avg, mae, rmse, acc_mean, all_preds, all_targets, all_errors
    
    return mae, rmse, acc_mean, all_preds, all_targets, all_errors

def compute_metrics(preds, targets):
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    
    errors = np.abs(preds - targets)
    acc_05 = np.mean(errors <= 0.05)
    acc_10 = np.mean(errors <= 0.10)
    acc_15 = np.mean(errors <= 0.15)
    
    component_metrics = {}
    for i, name in enumerate(['alpha', 'beta', 'unstructured']):
        component_metrics[name] = {
            'mae': mean_absolute_error(targets[:, i], preds[:, i]),
            'rmse': np.sqrt(mean_squared_error(targets[:, i], preds[:, i])),
            'acc_10': np.mean(np.abs(preds[:, i] - targets[:, i]) <= 0.10)
        }
    
    return {
        'mae': mae,
        'rmse': rmse,
        'acc_05': acc_05,
        'acc_10': acc_10,
        'acc_15': acc_15,
        'component': component_metrics
    }
