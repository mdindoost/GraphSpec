# src/training/trainer.py
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import time
from ..models.mlp import MLP
from ..models.gcn import GCN
from ..models.gat import GAT
from ..models.sage import GraphSAGE


class Trainer:
    """Unified trainer for all models."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, model_type='mlp',
                 lr=0.01, weight_decay=5e-4, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        
        # Create model
        if model_type == 'mlp':
            self.model = MLP(input_dim, hidden_dim, output_dim, dropout=0.5)
            self.use_graph = False
        elif model_type == 'gcn':
            self.model = GCN(input_dim, hidden_dim, output_dim, dropout=0.5)
            self.use_graph = True
        elif model_type == 'gat':
            self.model = GAT(input_dim, hidden_dim, output_dim, dropout=0.6)
            self.use_graph = True
        elif model_type == 'sage':
            self.model = GraphSAGE(input_dim, hidden_dim, output_dim, dropout=0.5)
            self.use_graph = True
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model = self.model.to(self.device)
        
        # IMPORTANT: Use proper optimizer settings
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def train_epoch(self, data, mask):
        """Single training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.use_graph:
            out = self.model(data.x, data.edge_index)
        else:
            out = self.model(data.x)
        
        loss = F.nll_loss(out[mask], data.y[mask])
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, data, mask):
        """Evaluate model."""
        self.model.eval()
        
        if self.use_graph:
            out = self.model(data.x, data.edge_index)
        else:
            out = self.model(data.x)
        
        pred = out.argmax(dim=1)
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        return acc, f1_micro, f1_macro
    
    def train(self, data, epochs=200, patience=50, verbose=True):  # Increased patience!
        """
        Full training loop.
        """
        data = data.to(self.device)
        
        best_val_acc = 0
        best_test_metrics = None
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            loss = self.train_epoch(data, data.train_mask)
            
            # Evaluate
            val_acc, val_f1_micro, val_f1_macro = self.evaluate(data, data.val_mask)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_metrics = self.evaluate(data, data.test_mask)
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
        
        train_time = time.time() - start_time
        
        test_acc, test_f1_micro, test_f1_macro = best_test_metrics
        
        if verbose:
            print(f"\nTest Accuracy:  {test_acc:.4f}")
            print(f"Test F1 (micro): {test_f1_micro:.4f}")
            print(f"Test F1 (macro): {test_f1_macro:.4f}")
            print(f"Training Time:   {train_time:.2f}s")
        
        return {
            'test_acc': test_acc,
            'test_f1_micro': test_f1_micro,
            'test_f1_macro': test_f1_macro,
            'train_time': train_time,
            'best_val_acc': best_val_acc
        }
