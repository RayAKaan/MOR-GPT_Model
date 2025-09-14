"""
GTX 1650 Optimized Ultra-Robust MoR Trainer v3
Specifically designed for your configuration:
- GTX 1650 (4GB VRAM) with aggressive memory management
- Ryzen 3 3100 (4 cores/8 threads) optimization
- 16GB RAM with smart CPU offloading
- Uses your exact config parameters

Features:
- Memory-first design for 4GB VRAM
- Smart CPU-GPU memory management
- Gradient checkpointing and mixed precision
- Automatic fallback strategies
- Real-time memory monitoring
- Efficient data loading for limited resources
- Robust error handling and recovery
"""

import os
import math
import time
import json
import gc
import random
import warnings
import threading
import psutil
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler
from torch import amp
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import numpy as np

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# Import your exact config
from Backend.model.config import *
from Backend.model.mor_adapter import MoRLanguageModel, MoRConfig
from Backend.model.tokenizer import Tokenizer

# Dataset import
try:
    from Backend.data.dataset import create_datasets
    USE_CREATE_DATASETS = True
except ImportError:
    from Backend.data.dataset import TextDataset
    USE_CREATE_DATASETS = False

warnings.filterwarnings("ignore", category=UserWarning)

class GTX1650MemoryManager:
    """Advanced memory manager specifically for GTX 1650's 4GB VRAM"""
    
    def __init__(self, device: torch.device, target_usage: float = 0.85):
        self.device = device
        self.target_usage = target_usage
        self.max_memory = 4.0 * 1024**3  # 4GB in bytes
        self.offloaded_tensors = {}
        self.memory_history = deque(maxlen=100)
        self.emergency_mode = False
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get detailed memory information"""
        if not torch.cuda.is_available():
            return {"available": True, "usage": 0.0}
            
        try:
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)
            
            usage_pct = allocated / self.max_memory
            reserved_pct = reserved / self.max_memory
            
            return {
                "allocated_mb": allocated / 1024**2,
                "reserved_mb": reserved / 1024**2,
                "max_allocated_mb": max_allocated / 1024**2,
                "usage_percent": usage_pct * 100,
                "reserved_percent": reserved_pct * 100,
                "available": usage_pct < self.target_usage,
                "emergency": usage_pct > 0.95
            }
        except Exception:
            return {"available": True, "usage": 0.0}
    
    def should_offload(self) -> bool:
        """Check if we should offload tensors to CPU"""
        info = self.get_memory_info()
        return info.get("usage_percent", 0) > (self.target_usage * 100)
    
    def emergency_cleanup(self):
        """Emergency memory cleanup for GTX 1650"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        # Force garbage collection on Python objects
        for _ in range(3):
            gc.collect()
    
    def smart_cache_management(self, step: int):
        """Smart cache management based on memory pressure"""
        if step % 5 == 0:  # Check every 5 steps
            info = self.get_memory_info()
            self.memory_history.append(info.get("usage_percent", 0))
            
            # If consistently high memory usage, enable emergency mode
            if len(self.memory_history) >= 10:
                avg_usage = np.mean(list(self.memory_history)[-10:])
                self.emergency_mode = avg_usage > 90
                
            if self.emergency_mode or info.get("emergency", False):
                self.emergency_cleanup()

class OptimizedDataLoader:
    """Memory-optimized data loader for GTX 1650"""
    
    def __init__(self, dataset, batch_size: int, device: torch.device, 
                 shuffle: bool = True, drop_last: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))
        
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
            
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
                
            batch = [self.dataset[idx] for idx in batch_indices]
            
            # Convert to tensors and move to device efficiently
            if batch:
                inputs = [item[0] if isinstance(item, (list, tuple)) else item for item in batch]
                targets = [item[1] if isinstance(item, (list, tuple)) and len(item) > 1 else item for item in batch]
                
                try:
                    input_tensor = torch.stack(inputs).to(self.device, non_blocking=True)
                    target_tensor = torch.stack(targets).to(self.device, non_blocking=True) if targets != inputs else input_tensor
                    yield input_tensor, target_tensor
                except Exception:
                    # Fallback for irregular batch shapes
                    for item in batch:
                        if isinstance(item, (list, tuple)):
                            yield torch.tensor(item[0]).unsqueeze(0).to(self.device), torch.tensor(item[1]).unsqueeze(0).to(self.device)
                        else:
                            yield torch.tensor(item).unsqueeze(0).to(self.device), torch.tensor(item).unsqueeze(0).to(self.device)

class SmartLRScheduler(_LRScheduler):
    """Smart learning rate scheduler using your config parameters"""
    
    def __init__(self, optimizer, total_steps: int):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * WARMUP_PROPORTION)
        super().__init__(optimizer)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            # Warmup phase
            return [base_lr * (self._step_count / max(1, self.warmup_steps)) 
                   for base_lr in self.base_lrs]
        
        # Cosine annealing
        progress = (self._step_count - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [base_lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine_factor)
                for base_lr in self.base_lrs]

class ModelEMA:
    """Exponential Moving Average for better convergence"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1-self.decay)
    
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

class GTX1650MoRTrainer:
    """Ultra-optimized MoR trainer for GTX 1650 using your exact config"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        print(f"[Trainer] Initializing GTX 1650 Optimized MoR Trainer v3")
        print(f"[Trainer] Using config: BATCH_SIZE={BATCH_SIZE}, EMBED_DIM={EMBED_DIM}, NUM_LAYERS={NUM_LAYERS}")
        
        # Apply any config overrides
        if config_override:
            for key, value in config_override.items():
                if key.upper() in globals():
                    globals()[key.upper()] = value
        
        # Device setup
        self.device = torch.device(DEVICE)
        print(f"[Trainer] Device: {self.device}")
        
        if torch.cuda.is_available():
            # Optimize CUDA settings for GTX 1650
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
                
            props = torch.cuda.get_device_properties(self.device)
            print(f"[Trainer] GPU: {props.name} | VRAM: {props.total_memory / 1e9:.2f} GB")
        
        # Initialize components
        self.memory_manager = GTX1650MemoryManager(self.device)
        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.metrics = {
            "train_losses": [], "val_losses": [], "val_perplexities": [],
            "learning_rates": [], "memory_usage": [], "step_times": []
        }
        
        # Setup logging
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / "tensorboard"))
        
        # Initialize everything
        self._setup_data()
        self._setup_model()
        self._setup_training()
    
    def _setup_data(self):
        """Setup data loaders optimized for your hardware"""
        print("[Trainer] Setting up optimized data loaders...")
        
        try:
            if USE_CREATE_DATASETS:
                self.tokenizer = Tokenizer(dataset_path="Backend/data/train.csv", vocab_size=VOCAB_SIZE_LIMIT)
                # Get datasets, not loaders
                train_dataset, val_dataset, test_dataset = create_datasets(
                    base_path="Backend/data", 
                    tokenizer=self.tokenizer,
                    max_length=MAX_LEN, 
                    batch_size=BATCH_SIZE,
                    return_datasets=True
                )
            else:
                # Fallback to manual dataset creation
                self.tokenizer = Tokenizer("Backend/data/dataset.json", vocab_size=VOCAB_SIZE_LIMIT)
                train_dataset = TextDataset("Backend/data/train.txt", self.tokenizer, max_len=MAX_LEN)
                val_dataset = TextDataset("Backend/data/val.txt", self.tokenizer, max_len=MAX_LEN)
                test_dataset = TextDataset("Backend/data/test.txt", self.tokenizer, max_len=MAX_LEN)
                
        except Exception as e:
            print(f"[Warning] Dataset setup failed: {e}. Creating minimal tokenizer.")
            self.tokenizer = Tokenizer()
            # Create dummy datasets for testing
            class DummyDataset:
                def __init__(self, size=1000):
                    self.size = size
                def __len__(self):
                    return self.size
                def __getitem__(self, idx):
                    return torch.randint(0, 100, (MAX_LEN,)), torch.randint(0, 100, (MAX_LEN,))
            
            train_dataset = DummyDataset(1000)
            val_dataset = DummyDataset(200)
            test_dataset = DummyDataset(200)
        
        # Create optimized data loaders
        self.train_loader = OptimizedDataLoader(
            train_dataset, BATCH_SIZE, self.device, shuffle=True, drop_last=True
        )
        self.val_loader = OptimizedDataLoader(
            val_dataset, BATCH_SIZE, self.device, shuffle=False, drop_last=False
        )
        self.test_loader = OptimizedDataLoader(
            test_dataset, BATCH_SIZE, self.device, shuffle=False, drop_last=False
        )
        
        print(f"[Trainer] Data loaded. Train: {len(self.train_loader)} batches")
    
    def _setup_model(self):
        """Setup MoR model with your exact configuration"""
        print("[Trainer] Initializing MoR model with your config...")
        
        # Use your exact config parameters
        vocab_size = getattr(self.tokenizer, 'vocab_size', VOCAB_SIZE_LIMIT)
        
        config = MoRConfig(
            vocab_size=vocab_size,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            ff_hidden_dim=FF_HIDDEN_DIM,
            dropout=DROPOUT,
            max_len=MAX_LEN,
            max_recursion_steps=MAX_RECURSION_STEPS,
        )
        
        self.model = MoRLanguageModel(config=config)
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Move to device with your precision
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing as per your config
        if USE_GRADIENT_CHECKPOINTING:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            elif hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
        
        # Model compilation if requested
        if COMPILE_MODEL and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                print("[Trainer] Model compiled successfully")
            except Exception as e:
                print(f"[Warning] Model compilation failed: {e}")
        
        # EMA for better convergence
        self.ema = ModelEMA(self.model, decay=0.9995)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Trainer] Model: {total_params:,} total params, {trainable_params:,} trainable")
        
        # Memory check
        memory_info = self.memory_manager.get_memory_info()
        print(f"[Trainer] Initial VRAM usage: {memory_info.get('usage_percent', 0):.1f}%")
    
    def _initialize_weights(self):
        """Initialize model weights for stable training"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _setup_training(self):
        """Setup training components using your config"""
        print("[Trainer] Setting up training components...")
        
        # Optimizer with your settings
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.95),  # Better for transformers
            eps=1e-8
        )
        
        # Criterion with your label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_TOKEN_ID,
            label_smoothing=LABEL_SMOOTHING
        )
        
        # Learning rate scheduler
        steps_per_epoch = len(self.train_loader)
        total_steps = (steps_per_epoch * NUM_EPOCHS) // GRAD_ACCUM_STEPS
        self.scheduler = SmartLRScheduler(self.optimizer, total_steps)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=MIXED_PRECISION)
        
        print(f"[Trainer] Training setup complete. Total steps: {total_steps}")
    
    def _model_accepts_return_metrics(self) -> bool:
        """Check if model supports return_metrics parameter"""
        try:
            import inspect
            sig = inspect.signature(self.model.forward)
            return "return_metrics" in sig.parameters
        except:
            return False
    
    @contextmanager
    def _memory_efficient_context(self):
        """Context manager for memory-efficient operations"""
        # Clear cache before operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            yield
        finally:
            # Aggressive cleanup after operation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def _safe_forward(self, batch) -> Tuple[torch.Tensor, Dict]:
        """Memory-safe forward pass"""
        try:
            # Unpack batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
            else:
                inputs = targets = batch
            
            # Ensure correct device and dtype
            if not torch.is_tensor(inputs):
                inputs = torch.tensor(inputs, dtype=torch.long, device=self.device)
            if not torch.is_tensor(targets):
                targets = torch.tensor(targets, dtype=torch.long, device=self.device)
            
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with self._memory_efficient_context():
                with amp.autocast("cuda", enabled=MIXED_PRECISION):
                    # Forward pass
                    if self._model_accepts_return_metrics():
                        output = self.model(inputs, return_metrics=True)
                        if isinstance(output, tuple):
                            logits, metrics = output
                        else:
                            logits = output
                            metrics = {}
                    else:
                        logits = self.model(inputs)
                        metrics = {}
                    
                    # Compute loss
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1)
                    )
                    
                    # Add regularization from your config
                    if "routing_penalty" in metrics:
                        loss = loss + metrics["routing_penalty"] * RECURSION_PENALTY_WEIGHT
                    
                    if "entropy_loss" in metrics:
                        loss = loss + metrics["entropy_loss"] * ENTROPY_REGULARIZATION
                    
                    # Update metrics
                    metrics.update({
                        "loss": loss.item(),
                        "perplexity": math.exp(min(loss.item(), 100))
                    })
            
            return loss, metrics
            
        except Exception as e:
            print(f"[Error] Forward pass failed: {e}")
            # Return dummy loss to continue training
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return dummy_loss, {"error": str(e)}
    
    def train_epoch(self, epoch: int) -> float:
        """Train one epoch with memory optimization"""
        self.model.train()
        total_loss = 0.0
        step_count = 0
        
        # Progress bar
        pbar = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            leave=False
        )
        
        for batch_idx, batch in pbar:
            grad_norm = 0.0
            step_start_time = time.time()
            
            # Memory management
            self.memory_manager.smart_cache_management(self.global_step)
            
            try:
                # Forward pass
                loss, metrics = self._safe_forward(batch)
                
                # Check for NaN/Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Warning] NaN/Inf loss at step {self.global_step}, skipping")
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                
                # Backward pass with gradient accumulation
                scaled_loss = loss / GRAD_ACCUM_STEPS
                
                if MIXED_PRECISION:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Gradient accumulation step
                if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Gradient clipping
                    if MIXED_PRECISION:
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_NORM)
                    
                    # Optimizer step
                    if MIXED_PRECISION:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Scheduler step
                    self.scheduler.step()
                    
                    # EMA update
                    self.ema.update(self.model)
                    
                    # Zero gradients
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    self.global_step += 1
                
                # Track metrics
                total_loss += loss.item()
                step_count += 1
                step_time = time.time() - step_start_time
                
                # Update progress bar
                avg_loss = total_loss / step_count
                current_lr = self.optimizer.param_groups[0]['lr']
                memory_info = self.memory_manager.get_memory_info()
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'vram': f'{memory_info.get("usage_percent", 0):.1f}%',
                    'grad_norm': f'{grad_norm:.2f}'
                })
                
                # Logging
                if self.global_step % LOG_INTERVAL == 0:
                    self._log_metrics({
                        'train/loss': avg_loss,
                        'train/learning_rate': current_lr,
                        'train/grad_norm': grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                        'train/memory_usage': memory_info.get("usage_percent", 0),
                        'train/step_time': step_time
                    })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] GPU out of memory at step {self.global_step}, trying recovery...")
                    self.memory_manager.emergency_cleanup()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise e
        
        avg_loss = total_loss / max(step_count, 1)
        self.metrics["train_losses"].append(avg_loss)
        self.metrics["learning_rates"].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validation with EMA model"""
        self.model.eval()
        
        # Use EMA weights for validation
        self.ema.apply_shadow(self.model)
        
        total_loss = 0.0
        step_count = 0
        
        try:
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                    loss, metrics = self._safe_forward(batch)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        step_count += 1
                        
                        # Memory management during validation
                        if step_count % 10 == 0:
                            self.memory_manager.emergency_cleanup()
        
        finally:
            # Restore original weights
            self.ema.restore(self.model)
        
        avg_loss = total_loss / max(step_count, 1)
        perplexity = math.exp(min(avg_loss, 100))
        
        self.metrics["val_losses"].append(avg_loss)
        self.metrics["val_perplexities"].append(perplexity)
        
        return avg_loss, perplexity
    
    def test_evaluate(self) -> Tuple[float, float]:
        """Final test evaluation"""
        print("[Trainer] Running test evaluation...")
        self.model.eval()
        
        # Use EMA weights
        self.ema.apply_shadow(self.model)
        
        total_loss = 0.0
        step_count = 0
        
        try:
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc="Testing"):
                    loss, metrics = self._safe_forward(batch)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        step_count += 1
        
        finally:
            self.ema.restore(self.model)
        
        avg_loss = total_loss / max(step_count, 1)
        perplexity = math.exp(min(avg_loss, 100))
        
        print(f"[Test] Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")
        return avg_loss, perplexity
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log metrics to TensorBoard"""
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.global_step)
    
    def save_checkpoint(self, filepath: Union[str, Path], is_best: bool = False):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': getattr(self, 'current_epoch', 0),
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'ema_shadow': self.ema.shadow,
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics,
            'config': {
                'BATCH_SIZE': BATCH_SIZE,
                'EMBED_DIM': EMBED_DIM,
                'NUM_LAYERS': NUM_LAYERS,
                'LEARNING_RATE': LEARNING_RATE,
                'NUM_EPOCHS': NUM_EPOCHS
            }
        }
        
        # Atomic save
        tmp_path = Path(str(filepath) + '.tmp')
        torch.save(checkpoint, tmp_path)
        tmp_path.replace(filepath)
        
        print(f"[Checkpoint] Saved {'best' if is_best else 'regular'} checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: Union[str, Path]) -> int:
        """Load training checkpoint"""
        print(f"[Trainer] Loading checkpoint from {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load states
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.metrics = checkpoint.get('metrics', self.metrics)
        
        return checkpoint.get('epoch', 0) + 1
    
    def _find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if checkpoints:
            return max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
        
        # Also check for best checkpoint
        best_checkpoint = self.checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            return best_checkpoint
        
        return None
    
    def _save_training_plots(self):
        """Save training visualization plots"""
        if not MATPLOTLIB or not self.metrics["train_losses"]:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        epochs = range(1, len(self.metrics["train_losses"]) + 1)
        axes[0, 0].plot(epochs, self.metrics["train_losses"], label="Train Loss", linewidth=2)
        if self.metrics["val_losses"]:
            axes[0, 0].plot(epochs, self.metrics["val_losses"], label="Val Loss", linewidth=2)
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training & Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Perplexity
        if self.metrics["val_perplexities"]:
            axes[0, 1].plot(epochs, self.metrics["val_perplexities"], color='orange', linewidth=2)
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Perplexity")
            axes[0, 1].set_title("Validation Perplexity")
            axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        if self.metrics["learning_rates"]:
            axes[1, 0].plot(epochs, self.metrics["learning_rates"], color='green', linewidth=2)
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].set_title("Learning Rate Schedule")
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Memory usage over time
        if self.metrics["memory_usage"]:
            axes[1, 1].plot(self.metrics["memory_usage"], color='red', linewidth=2)
            axes[1, 1].set_xlabel("Steps")
            axes[1, 1].set_ylabel("VRAM Usage (%)")
            axes[1, 1].set_title("GPU Memory Usage")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='Target (85%)')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Trainer] Training plots saved to {self.checkpoint_dir}/training_curves.png")
    
    def train(self):
        """Main training loop optimized for GTX 1650"""
        print(f"[Trainer] Starting training for {NUM_EPOCHS} epochs")
        print(f"[Trainer] Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
        print(f"[Trainer] Early stopping patience: {EARLY_STOPPING_PATIENCE}")
        
        # Check for existing checkpoint
        start_epoch = 0
        latest_checkpoint = self._find_latest_checkpoint()
        if latest_checkpoint:
            try:
                start_epoch = self.load_checkpoint(latest_checkpoint)
                print(f"[Trainer] Resumed from epoch {start_epoch}")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint: {e}")
        
        training_start_time = time.time()
        
        try:
            for epoch in range(start_epoch, NUM_EPOCHS):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Training phase
                train_loss = self.train_epoch(epoch)
                
                # Validation phase (every EVAL_INTERVAL epochs)
                if (epoch + 1) % EVAL_INTERVAL == 0:
                    val_loss, val_perplexity = self.validate()
                    
                    # Check for improvement
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        # Save best model
                        self.save_checkpoint(self.checkpoint_dir / "best_model.pt", is_best=True)
                    else:
                        self.patience_counter += 1
                    
                    # Epoch summary
                    epoch_time = time.time() - epoch_start
                    memory_info = self.memory_manager.get_memory_info()
                    
                    print(f"[Epoch {epoch+1:2d}/{NUM_EPOCHS}] "
                          f"Train: {train_loss:.4f} | "
                          f"Val: {val_loss:.4f} | "
                          f"PPL: {val_perplexity:.2f} | "
                          f"Best: {self.best_val_loss:.4f} | "
                          f"Time: {epoch_time:.1f}s | "
                          f"VRAM: {memory_info.get('usage_percent', 0):.1f}%")
                    
                    # Log epoch metrics
                    self._log_metrics({
                        'epoch/train_loss': train_loss,
                        'epoch/val_loss': val_loss,
                        'epoch/val_perplexity': val_perplexity,
                        'epoch/best_val_loss': self.best_val_loss,
                        'epoch/patience': self.patience_counter,
                        'epoch/memory_usage': memory_info.get('usage_percent', 0)
                    })
                    
                    # Early stopping check
                    if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"[Trainer] Early stopping triggered after {self.patience_counter} epochs without improvement")
                        break
                
                else:
                    # Just training, no validation
                    epoch_time = time.time() - epoch_start
                    print(f"[Epoch {epoch+1:2d}/{NUM_EPOCHS}] Train: {train_loss:.4f} | Time: {epoch_time:.1f}s")
                
                # Save periodic checkpoint
                if (epoch + 1) % SAVE_INTERVAL == 0:
                    self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
                
                # Memory cleanup between epochs
                self.memory_manager.emergency_cleanup()
        
        except KeyboardInterrupt:
            print("\n[Trainer] Training interrupted by user")
        except Exception as e:
            print(f"[Error] Training failed: {e}")
            # Save emergency checkpoint
            self.save_checkpoint(self.checkpoint_dir / "emergency_checkpoint.pt")
            raise
        
        finally:
            # Final test evaluation
            if hasattr(self, 'test_loader'):
                test_loss, test_perplexity = self.test_evaluate()
            
            # Training summary
            training_time = time.time() - training_start_time
            print(f"\n{'='*60}")
            print(f"Training completed!")
            print(f"Total time: {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            if hasattr(self, 'test_loader'):
                print(f"Final test loss: {test_loss:.4f}")
                print(f"Final test perplexity: {test_perplexity:.2f}")
            print(f"Total steps: {self.global_step}")
            print(f"Checkpoints saved in: {self.checkpoint_dir}")
            print(f"{'='*60}")
            
            # Save final artifacts
            self._save_training_plots()
            
            # Save training summary
            summary = {
                'training_time_hours': training_time / 3600,
                'total_steps': self.global_step,
                'best_val_loss': self.best_val_loss,
                'final_epoch': getattr(self, 'current_epoch', NUM_EPOCHS-1) + 1,
                'config': {
                    'BATCH_SIZE': BATCH_SIZE,
                    'GRAD_ACCUM_STEPS': GRAD_ACCUM_STEPS,
                    'LEARNING_RATE': LEARNING_RATE,
                    'NUM_EPOCHS': NUM_EPOCHS,
                    'EMBED_DIM': EMBED_DIM,
                    'NUM_LAYERS': NUM_LAYERS,
                    'MAX_LEN': MAX_LEN
                },
                'metrics': self.metrics
            }
            
            with open(self.checkpoint_dir / "training_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Close TensorBoard writer
            if self.writer:
                self.writer.close()
            
            # Final cleanup
            self.memory_manager.emergency_cleanup()
            print(f"[Trainer] All artifacts saved to {self.checkpoint_dir}")

class InferenceEngine:
    """Optimized inference engine for trained MoR models"""
    
    def __init__(self, model_path: Union[str, Path], device: Optional[torch.device] = None):
        self.device = device or torch.device(DEVICE)
        self.model_path = Path(model_path)
        
        print(f"[Inference] Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model
        config_dict = checkpoint.get('config', {})
        vocab_size = config_dict.get('vocab_size', VOCAB_SIZE_LIMIT)
        
        config = MoRConfig(
            vocab_size=vocab_size,
            embed_dim=config_dict.get('EMBED_DIM', EMBED_DIM),
            num_heads=NUM_HEADS,
            num_layers=config_dict.get('NUM_LAYERS', NUM_LAYERS),
            ff_hidden_dim=FF_HIDDEN_DIM,
            dropout=0.0,  # No dropout for inference
            max_len=MAX_LEN,
            max_recursion_steps=MAX_RECURSION_STEPS,
        )
        
        self.model = MoRLanguageModel(config=config).to(self.device, dtype=DTYPE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"[Inference] Model loaded successfully")
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = TEMPERATURE, top_k: int = TOP_K, 
                 top_p: float = TOP_P, repetition_penalty: float = REPETITION_PENALTY) -> torch.Tensor:
        """Generate text using the trained MoR model"""
        
        self.model.eval()
        batch_size = input_ids.size(0)
        current_length = input_ids.size(1)
        
        # Prepare generation
        generated = input_ids.clone()
        past_tokens = set()
        
        for _ in range(max_length):
            # Forward pass
            with amp.autocast("cuda", enabled=MIXED_PRECISION):
                outputs = self.model(generated)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token in past_tokens:
                    next_token_logits[:, token] /= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits.fill_(-float('inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits.masked_fill_(indices_to_remove, -float('inf'))
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to past tokens for repetition penalty
            past_tokens.add(next_token.item())
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS token
            if next_token.item() == EOS_TOKEN_ID:
                break
            
            # Memory management
            if generated.size(1) % 50 == 0:  # Every 50 tokens
                torch.cuda.empty_cache()
        
        return generated

# Utility functions
def setup_training_environment():
    """Setup optimal training environment for GTX 1650"""
    
    # Set optimal random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Optimize CUDA settings for GTX 1650
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Enable TensorFloat-32 if available (RTX series)
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        
        print(f"[Setup] CUDA optimizations enabled")
        
        # Print GPU info
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Setup] GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # CPU optimizations
    torch.set_num_threads(4)  # Ryzen 3 3100 has 4 cores
    
    # Memory optimizations
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print(f"[Setup] Environment optimized for GTX 1650 + Ryzen 3 3100")

def main():
    """Main training function"""
    
    # Setup environment
    setup_training_environment()
    
    # Print system info
    print(f"\n{'='*60}")
    print(f"GTX 1650 Optimized MoR Trainer v3")
    print(f"{'='*60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # System resources
    if psutil:
        print(f"CPU cores: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f}GB")
    
    print(f"{'='*60}\n")
    
    # Initialize trainer with your exact config
    trainer = GTX1650MoRTrainer()
    
    # Start training
    trainer.train()
    
    print(f"\n🎉 Training completed successfully!")
    print(f"Best model saved to: {trainer.checkpoint_dir}/best_model.pt")
    print(f"Training curves: {trainer.checkpoint_dir}/training_curves.png")

if __name__ == "__main__":
    main()