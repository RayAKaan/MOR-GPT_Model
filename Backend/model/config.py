# Backend/model/config.py — Robust MOR Configuration System v3

import os
import json
import yaml
import torch
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum
import warnings
from contextlib import contextmanager


# === Enums for Better Type Safety ===

class DeviceType(Enum):
    AUTO = "auto"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"

class PrecisionType(Enum):
    FLOAT32 = torch.float32
    FLOAT16 = torch.float16
    BFLOAT16 = torch.bfloat16

class SchedulerType(Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"

class OptimizerType(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    SGD = "sgd"
    ADAGRAD = "adagrad"


# === Configuration Dataclasses ===

@dataclass
class DeviceConfig:
    """Device and precision configuration."""
    device_type: DeviceType = DeviceType.AUTO
    precision: PrecisionType = PrecisionType.FLOAT16
    cuda_device_id: int = 0
    enable_tf32: bool = True  # For better performance on RTX cards
    enable_flash_attention: bool = False  # Disable for GTX 1650
    memory_fraction: float = 0.9  # Use 90% of GPU memory
    
    def get_device(self) -> torch.device:
        """Get the actual torch device."""
        if self.device_type == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.cuda_device_id}")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.device_type.value)
    
    def setup_device_optimizations(self):
        """Setup device-specific optimizations."""
        device = self.get_device()
        
        if device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction, device.index)
            
            if self.enable_tf32 and torch.cuda.get_device_capability(device)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            torch.cuda.empty_cache()
        
        return device


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    embed_dim: int = 192
    num_heads: int = 4
    num_layers: int = 3
    ff_hidden_dim: int = 768
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    use_bias: bool = True
    max_position_embeddings: int = 512
    position_encoding_type: str = "learned"
    use_gradient_checkpointing: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = True
    
    def validate(self):
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.embed_dim > 0
        assert self.num_heads > 0
        assert self.num_layers > 0


@dataclass
class MORConfig:
    """Mixture-of-Routers (MOR) specific configuration."""
    max_recursion_steps: int = 3
    min_recursion_steps: int = 1
    recursion_penalty_weight: float = 0.005
    router_temperature: float = 0.7
    hard_routing_threshold: float = 0.9
    load_balancing_weight: float = 0.01
    router_noise_std: float = 0.0
    num_experts: int = 3
    top_k_experts: int = 2
    expert_capacity_factor: float = 1.0
    routing_strategy: str = "top_k"
    auxiliary_loss_weight: float = 0.01
    residual_connection_strength: float = 1.0
    expert_dropout: float = 0.1
    
    def validate(self):
        assert self.max_recursion_steps >= self.min_recursion_steps
        assert self.top_k_experts <= self.num_experts
        assert 0.0 <= self.router_temperature <= 2.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_epochs: int = 20
    max_steps: Optional[int] = None
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 2e-5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    scheduler_type: SchedulerType = SchedulerType.COSINE
    warmup_steps: Optional[int] = None
    warmup_proportion: float = 0.2
    min_lr_ratio: float = 0.01
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.1
    dropout: float = 0.1
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 1e-4
    mixed_precision: bool = True
    loss_scale: str = "dynamic"
    
    def get_effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def get_total_steps(self, dataset_size: int) -> int:
        steps_per_epoch = dataset_size // self.get_effective_batch_size()
        if self.max_steps:
            return min(self.max_steps, steps_per_epoch * self.num_epochs)
        return steps_per_epoch * self.num_epochs
    
    def get_warmup_steps(self, total_steps: int) -> int:
        if self.warmup_steps:
            return self.warmup_steps
        return int(total_steps * self.warmup_proportion)


@dataclass
class DataConfig:
    """Data processing configuration."""
    max_length: int = 192
    min_length: int = 5
    pad_to_multiple_of: int = 8
    vocab_size_limit: int = 3000
    cache_dir: Optional[str] = None
    num_workers: int = 2
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    enable_data_augmentation: bool = False
    augmentation_probability: float = 0.1
    
    def validate(self):
        assert self.max_length > self.min_length
        assert self.vocab_size_limit > 100


@dataclass
class InferenceConfig:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    length_penalty: float = 1.0
    max_new_tokens: int = 100
    min_new_tokens: int = 1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    batch_size: int = 1
    
    def validate(self):
        assert 0.0 < self.temperature <= 2.0
        assert 0.0 <= self.top_p <= 1.0


@dataclass
class MonitoringConfig:
    log_level: str = "INFO"
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 5
    checkpoint_dir: str = "checkpoints"
    save_total_limit: int = 3
    save_safetensors: bool = True
    track_gpu_memory: bool = True
    track_gradients: bool = False
    track_weights: bool = False
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: Optional[str] = None
    enable_profiling: bool = False
    profile_memory: bool = False
    
    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


@dataclass
class OptimizationConfig:
    compile_model: bool = False
    compile_mode: str = "default"
    use_gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = True
    empty_cache_steps: int = 100
    use_fused_adamw: bool = True
    use_scaled_dot_product_attention: bool = True
    use_ddp: bool = False
    find_unused_parameters: bool = False
    use_triton_optimizations: bool = False
    use_flash_attention_2: bool = False


@dataclass
class MORTrainingConfig:
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mor: MORConfig = field(default_factory=MORConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    seed: int = 42
    deterministic: bool = False
    experiment_name: str = "mor_training"
    special_tokens: Dict[str, str] = field(default_factory=lambda: {
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "mask_token": "<mask>",
        "sep_token": "<sep>"
    })
    
    def __post_init__(self):
        self.validate_all()
        self.setup_determinism()
        self.apply_device_optimizations()
        self.print_config_summary()
    
    def validate_all(self):
        try:
            self.model.validate()
            self.mor.validate()
            self.data.validate()
            self.inference.validate()
        except AssertionError as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    def setup_determinism(self):
        if self.deterministic:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(self.seed)
    
    def apply_device_optimizations(self):
        self.device.setup_device_optimizations()
    
    def get_model_size_estimate(self) -> Dict[str, float]:
        embed_params = self.data.vocab_size_limit * self.model.embed_dim
        attention_params = (4 * self.model.embed_dim * self.model.embed_dim + self.model.embed_dim) * self.model.num_layers
        ffn_params = (2 * self.model.embed_dim * self.model.ff_hidden_dim + self.model.ff_hidden_dim + self.model.embed_dim) * self.model.num_layers
        ln_params = 2 * self.model.embed_dim * (self.model.num_layers + 1)
        total_params = embed_params + attention_params + ffn_params + ln_params
        param_memory = total_params * 4 / (1024 * 1024)
        gradient_memory = param_memory
        optimizer_memory = param_memory * 2
        activation_memory = (self.training.batch_size * self.data.max_length * self.model.embed_dim * 4 * self.model.num_layers / (1024 * 1024))
        total_training_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
        return {
            "total_parameters": total_params,
            "parameter_memory_mb": param_memory,
            "total_training_memory_mb": total_training_memory,
            "activation_memory_mb": activation_memory,
            "optimizer_memory_mb": optimizer_memory
        }
    
    def print_config_summary(self):
        device = self.device.get_device()
        memory_info = self.get_model_size_estimate()
        print("=" * 80)
        print("🚀 MOR Training Configuration Summary")
        print("=" * 80)
        print(f"📱 Device & Precision:")
        print(f"   Device: {device} | Precision: {self.device.precision.value}")
        print(f"   Mixed Precision: {self.training.mixed_precision}")
        print(f"   Gradient Checkpointing: {self.optimization.use_gradient_checkpointing}")
        print(f"\n🏗️  Model Architecture:")
        print(f"   Embedding: {self.model.embed_dim}d | Heads: {self.model.num_heads}")
        print(f"   Layers: {self.model.num_layers} | FFN Hidden: {self.model.ff_hidden_dim}")
        print(f"   Max Length: {self.data.max_length} | Vocab Size: {self.data.vocab_size_limit}")
        print(f"\n🔄 MOR Configuration:")
        print(f"   Recursion Steps: {self.mor.min_recursion_steps}-{self.mor.max_recursion_steps}")
        print(f"   Experts: {self.mor.num_experts} | Top-K: {self.mor.top_k_experts}")
        print(f"   Router Temperature: {self.mor.router_temperature}")
        print(f"\n🎯 Training Setup:")
        print(f"   Batch Size: {self.training.batch_size} (x{self.training.gradient_accumulation_steps} accum)")
        print(f"   Effective Batch: {self.training.get_effective_batch_size()}")
        print(f"   Learning Rate: {self.training.learning_rate} | Weight Decay: {self.training.weight_decay}")
        print(f"   Epochs: {self.training.num_epochs} | Early Stop: {self.training.early_stopping_patience}")
        print(f"\n💾 Memory Estimates:")
        print(f"   Parameters: {memory_info['total_parameters']:,.0f} ({memory_info['parameter_memory_mb']:.1f} MB)")
        print(f"   Training Memory: ~{memory_info['total_training_memory_mb']:.1f} MB")
        print(f"\n⚡ Optimizations:")
        print(f"   Model Compilation: {self.optimization.compile_model}")
        print(f"   Fused AdamW: {self.optimization.use_fused_adamw}")
        print(f"   Flash Attention: {self.device.enable_flash_attention}")
        print("=" * 80)
        if device.type == "cuda" and memory_info['total_training_memory_mb'] > 3500:
            warnings.warn(f"⚠️  Estimated memory usage ({memory_info['total_training_memory_mb']:.1f} MB) might exceed GTX 1650 capacity.")


# === Automatic Device Logging ===
def log_device_info(device_config: DeviceConfig):
    device = device_config.get_device()
    print("=" * 80)
    print("💻 Device Info")
    print("=" * 80)
    print(f"Device Type      : {device.type}")
    if device.type == "cuda":
        print(f"CUDA Device ID   : {device.index}")
        print(f"Memory Fraction  : {device_config.memory_fraction}")
        print(f"TF32 Enabled     : {device_config.enable_tf32 and torch.cuda.get_device_capability(device)[0] >= 8}")
        print(f"Flash Attention  : {device_config.enable_flash_attention}")
        print(f"CUDA Capability  : {torch.cuda.get_device_capability(device)}")
        print(f"CUDA Memory Alloc: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
        print(f"CUDA Memory Cached: {torch.cuda.memory_reserved(device)/1024**2:.1f} MB")
    elif device.type == "mps":
        print("MPS Device detected (Apple Silicon)")
    else:
        print("CPU Device detected")
    print("=" * 80)

_original_post_init = MORTrainingConfig.__post_init__
def _enhanced_post_init(self):
    _original_post_init(self)
    log_device_info(self.device)
MORTrainingConfig.__post_init__ = _enhanced_post_init


# === Presets ===
def get_gtx_1650_config() -> MORTrainingConfig:
    return MORTrainingConfig(
        device=DeviceConfig(device_type=DeviceType.AUTO, precision=PrecisionType.FLOAT16, enable_flash_attention=False, memory_fraction=0.85),
        model=ModelConfig(embed_dim=192, num_heads=4, num_layers=3, ff_hidden_dim=768),
        mor=MORConfig(max_recursion_steps=3, min_recursion_steps=1, num_experts=3, top_k_experts=2),
        training=TrainingConfig(batch_size=4, gradient_accumulation_steps=2, learning_rate=2e-5, num_epochs=20, mixed_precision=True),
        data=DataConfig(max_length=192, vocab_size_limit=3000),
        monitoring=MonitoringConfig(track_gpu_memory=True, save_total_limit=2),
        optimization=OptimizationConfig(use_gradient_checkpointing=True, compile_model=False)
    )

def get_high_end_config() -> MORTrainingConfig:
    return MORTrainingConfig(
        device=DeviceConfig(device_type=DeviceType.AUTO, precision=PrecisionType.BFLOAT16, enable_flash_attention=True, memory_fraction=0.95),
        model=ModelConfig(embed_dim=512, num_heads=8, num_layers=12, ff_hidden_dim=2048),
        mor=MORConfig(max_recursion_steps=5, num_experts=8, top_k_experts=3),
        training=TrainingConfig(batch_size=16, gradient_accumulation_steps=1, learning_rate=5e-5, num_epochs=50),
        data=DataConfig(max_length=512, vocab_size_limit=50000),
        optimization=OptimizationConfig(compile_model=True, use_triton_optimizations=True, use_flash_attention_2=True)
    )

def get_cpu_config() -> MORTrainingConfig:
    return MORTrainingConfig(
        device=DeviceConfig(device_type=DeviceType.CPU, precision=PrecisionType.FLOAT32),
        model=ModelConfig(embed_dim=128, num_heads=2, num_layers=2, ff_hidden_dim=256),
        training=TrainingConfig(batch_size=1, gradient_accumulation_steps=8, mixed_precision=False, num_epochs=10),
        data=DataConfig(max_length=128, vocab_size_limit=1000, num_workers=0)
    )


# === Legacy Support ===
config = get_gtx_1650_config()  # Default for backward compatibility
