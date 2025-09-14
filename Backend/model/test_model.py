# 📦 Backend/model/test_model.py — MoR Model Tester & Generator (Optimized)

import torch
import torch.nn.functional as F
from pathlib import Path
import re
import json
import time
import gc
from typing import Optional, Dict, Any

# Import your modules
from .tokenizer import Tokenizer
from .mor_adapter import MoRLanguageModel  # Your optimized MoR model
from .config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN_DIM, DROPOUT, MAX_RECURSION_STEPS, MAX_LEN, ROUTER_TEMPERATURE, ROUTER_NOISE_STD, RESIDUAL_CONNECTION_STRENGTH, USE_GRADIENT_CHECKPOINTING, RECURSION_PENALTY_WEIGHT, ENTROPY_REGULARIZATION

# ====== DEVICE & SETUP ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_properties(0).name}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ====== PATHS ======
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
DATASET_PATH = BASE_DIR / "data" / "dataset.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def _map_checkpoint_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Maps old GPT-style checkpoint keys to the new MoR model keys."""
    
    print("Mapping old checkpoint keys to new model structure...")
    
    key_map = [
        (r'^token_embed\.weight$', 'token_emb.weight'),
        (r'^pos_embed\.weight$', 'pos_emb.weight'),
        (r'^ln_f\.([\w]+)$', r'norm.\1'),
        (r'^head\.weight$', 'lm_head.weight'),
        (r'^blocks\.([\d]+)\.ln1\.([\w]+)$', r'recursive_stack.layers.\1.norm1.\2'),
        (r'^blocks\.([\d]+)\.ln2\.([\w]+)$', r'recursive_stack.layers.\1.norm2.\2'),
        (r'^blocks\.([\d]+)\.attn\.qkv_proj\.weight$', r'recursive_stack.layers.\1.attn.in_proj_weight'),
        (r'^blocks\.([\d]+)\.attn\.out_proj\.([\w]+)$', r'recursive_stack.layers.\1.attn.out_proj.\2'),
        (r'^blocks\.([\d]+)\.ff\.net\.0\.([\w]+)$', r'recursive_stack.layers.\1.ff.0.\2'),
        (r'^blocks\.([\d]+)\.ff\.net\.3\.([\w]+)$', r'recursive_stack.layers.\1.ff.3.\2'),
    ]
    
    new_state_dict = {}
    for old_key, value in state_dict.items():
        if 'attn.mask' in old_key:
            continue

        new_key = old_key
        for pattern, replacement in key_map:
            if re.match(pattern, old_key):
                new_key = re.sub(pattern, replacement, old_key)
                break
        
        new_state_dict[new_key] = value
        
    print(f"Key mapping complete. {len(new_state_dict)} keys processed.")
    return new_state_dict

class MoRModelTester:
    """Optimized MoR Model Tester with Advanced Generation"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = DEVICE
        self.model_config = {}
        
        # Generation presets
        self.generation_presets = {
            'creative': {
                'temperature': 0.9,
                'top_k': 50,
                'top_p': 0.95,
                'repetition_penalty': 1.1
            },
            'balanced': {
                'temperature': 0.8,
                'top_k': 40,
                'top_p': 0.9,
                'repetition_penalty': 1.15
            },
            'focused': {
                'temperature': 0.6,
                'top_k': 25,
                'top_p': 0.8,
                'repetition_penalty': 1.2
            },
            'precise': {
                'temperature': 0.3,
                'top_k': 10,
                'top_p': 0.7,
                'repetition_penalty': 1.25
            }
        }
        
        self.setup()
    
    def setup(self):
        """Initialize tokenizer and model"""
        print("Setting up MoR Model Tester...")
        
        # Load tokenizer
        self.load_tokenizer()
        
        # Load model
        self.load_model()
        
        print("Setup complete!")
    
    def load_tokenizer(self):
        """Load or build tokenizer"""
        if TOKENIZER_PATH.exists():
            print(f"Loading tokenizer from {TOKENIZER_PATH}")
            self.tokenizer = Tokenizer.load(TOKENIZER_PATH)
            print(f"Tokenizer loaded (vocab size: {self.tokenizer.get_vocab_size()})")
        else:
            print("Building tokenizer from dataset...")
            self.tokenizer = Tokenizer(
                dataset_path=str(DATASET_PATH), 
                vocab_size=3000, 
                verbose=True
            )
            self.tokenizer.save(str(TOKENIZER_PATH))
            print(f"Tokenizer built and saved (vocab size: {self.tokenizer.vocab_size})")
    
    def get_latest_checkpoint(self) -> Path:
        """Find the latest checkpoint"""
        # Look for MoR checkpoints first
        mor_files = list(CHECKPOINT_DIR.rglob("mor_*.pt"))
        if mor_files:
            # Prefer best model
            best_file = CHECKPOINT_DIR / "mor_best.pt"
            if best_file.exists():
                return best_file
            
            # Otherwise get latest epoch
            def extract_epoch(f):
                match = re.search(r"mor_epoch_(\d+).pt", str(f))
                return int(match.group(1)) if match else -1
            
            return max(mor_files, key=extract_epoch)
        
        # Fallback to old GPT checkpoints
        gpt_files = list(CHECKPOINT_DIR.rglob("gpt_epoch*.pt"))
        if gpt_files:
            def extract_epoch(f):
                match = re.search(r"gpt_epoch(\d+).pt", str(f))
                return int(match.group(1)) if match else -1
            return max(gpt_files, key=extract_epoch)
        
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    
    def load_model(self):
        """Load the trained MoR model"""
        checkpoint_path = self.get_latest_checkpoint()
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint and determine its type
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict_for_check = checkpoint.get('model_state_dict', checkpoint)
        is_old_gpt_ckpt = any("blocks" in k for k in state_dict_for_check.keys())

        # Get model configuration
        if 'config' in checkpoint:
            self.model_config = checkpoint['config']
            print("Using saved model configuration")
        elif is_old_gpt_ckpt:
            print("Old GPT checkpoint detected. Rebuilding legacy configuration.")
            # This configuration is derived from the gpt_epoch35.pt checkpoint's tensor shapes
            self.model_config = {
                'vocab_size': 2853,
                'embed_dim': 160,
                'num_heads': 8,       # Assumed, but doesn't affect shape of combined QKV
                'num_layers': 6,      # Inferred from 'blocks.5' in checkpoint keys
                'ff_hidden_dim': 640, # Inferred from feed-forward layer shapes
                'dropout': 0.1,       # A reasonable default
                'max_len': 128,
                'max_recursion_steps': 3, # Fallback to a default value
                'router_temperature': 1.0,
                'router_noise_std': 0.0,
                'residual_connection_strength': 1.0,
                'use_gradient_checkpointing': False,
                'recursion_penalty_weight': 0.01
            }
            # Add missing keys that the new model expects in its config
            self.model_config['max_steps'] = self.model_config['max_recursion_steps']

        else:
            # Fallback to current config from config.py for new models without a config dict
            print("No config found in checkpoint. Using fallback from config.py.")
            self.model_config = {
                'vocab_size': self.tokenizer.vocab_size,
                'embed_dim': EMBED_DIM,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'ff_hidden_dim': FF_HIDDEN_DIM,
                'dropout': DROPOUT,
                'max_steps': MAX_RECURSION_STEPS,
                'max_len': MAX_LEN,
                'max_recursion_steps': MAX_RECURSION_STEPS,
                'router_temperature': ROUTER_TEMPERATURE,
                'router_noise_std': ROUTER_NOISE_STD,
                'residual_connection_strength': RESIDUAL_CONNECTION_STRENGTH,
                'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
                'recursion_penalty_weight': RECURSION_PENALTY_WEIGHT,
                'entropy_regularization_weight': ENTROPY_REGULARIZATION,
                'use_kv_cache': True, # Default to True for testing
                'use_expert_choice': False, # Default to False for testing
                'expert_capacity_factor': 1.2, # Default value
                'router_detach_inputs': True, # Default value
                'kv_cache_device': "cuda" # Default value
            }
        
        # Convert dict to object for MoRLanguageModel
        class ConfigObject:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        config_obj = ConfigObject(self.model_config)
        self.model = MoRLanguageModel(config=config_obj).to(self.device)

        # Load state dict, mapping old keys if necessary
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        if is_old_gpt_ckpt:
            state_dict = _map_checkpoint_keys(state_dict)
            # Load with strict=False because router/MoR specific keys are missing
            self.model.load_state_dict(state_dict, strict=False)
            print("Mapped state_dict loaded (non-strict).")
        else:
            self.model.load_state_dict(state_dict)
            print("State_dict loaded.")
        
        self.model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {total_params:,} parameters")
        print(f"Architecture: {self.model_config['embed_dim']}d, {self.model_config['num_heads']}h, {self.model_config['num_layers']}L")
        print(f"Max recursion steps: {self.model_config.get('max_steps', 'N/A')}")
        
        # Print metrics if available
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"Training metrics:")
            loss = metrics.get('total_loss')
            print(f"   Loss: {loss:.4f}" if isinstance(loss, float) else f"   Loss: {loss or 'N/A'}")
            perplexity = metrics.get('perplexity')
            print(f"   Perplexity: {perplexity:.2f}" if isinstance(perplexity, float) else f"   Perplexity: {perplexity or 'N/A'}")
            accuracy = metrics.get('accuracy')
            print(f"   Accuracy: {accuracy:.3f}" if isinstance(accuracy, float) else f"   Accuracy: {accuracy or 'N/A'}")
    
    @torch.no_grad()
    def generate_text(self, 
                     prompt: str, 
                     max_new_tokens: int = 50,
                     temperature: float = 0.8,
                     top_k: Optional[int] = 40,
                     top_p: Optional[float] = 0.9,
                     repetition_penalty: float = 1.15,
                     do_sample: bool = True,
                     show_progress: bool = False,
                     show_routing: bool = False) -> Dict[str, Any]:
        """Advanced text generation with MoR model"""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first!")
        
        start_time = time.time()
        
        # Encode prompt
        token_ids = self.tokenizer.encode(prompt)
        
        # Ensure we don't exceed context length
        max_context = MAX_LEN - max_new_tokens
        if len(token_ids) > max_context:
            token_ids = token_ids[-max_context:]
            print(f"Prompt truncated to {max_context} tokens")
        
        # Convert to tensor
        input_ids = torch.tensor([token_ids], device=self.device)
        original_length = input_ids.size(1)
        
        # Track generation stats
        routing_stats = []
        token_probs = []
        
        # Generation loop
        for step in range(max_new_tokens):
            # Forward pass
            output = self.model(input_ids)
            if len(output) == 3:
                logits, entropy_loss, routing_penalty = output
            else:
                logits, metrics = output
                entropy_loss = metrics.get('entropy_loss', torch.tensor(0.0))
                routing_penalty = metrics.get('routing_penalty', torch.tensor(0.0))
            
            next_token_logits = logits[0, -1, :]  # Get logits for last token
            
            # Track routing stats
            if show_routing:
                penalty_weight = self.model.config.recursion_penalty_weight
                avg_steps = routing_penalty.item() / penalty_weight if penalty_weight > 0 else 0.0
                routing_stats.append({
                    'step': step,
                    'avg_recursion_steps': avg_steps,
                    'entropy_loss': entropy_loss.item()
                })
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_token_logits[token_id] < 0:
                        next_token_logits[token_id] *= repetition_penalty
                    else:
                        next_token_logits[token_id] /= repetition_penalty
            
            # Convert to probabilities
            probs = F.softmax(next_token_logits, dim=-1) + 1e-8 # Add epsilon
            probs = probs / probs.sum(dim=-1, keepdim=True) # Re-normalize
            
            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                probs = torch.zeros_like(probs)
                probs.scatter_(0, top_k_indices, top_k_probs)
                probs = probs / probs.sum()
            
            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=0)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum()
            
            # Sample next token
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Track token probability
            token_probs.append(probs[next_token].item())
            
            # Add to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # Check for EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                print("EOS token reached")
                break
            
            # Progress indicator
            if show_progress and (step + 1) % 10 == 0:
                print(f"Generated {step + 1}/{max_new_tokens} tokens...")
            
            # Memory management for long sequences
            if input_ids.size(1) > self.model.config.max_len:
                # Keep recent context
                keep_tokens = self.model.config.max_len // 2
                input_ids = torch.cat([
                    input_ids[:, :10],  # Keep first few tokens for context
                    input_ids[:, -keep_tokens:]
                ], dim=1)
        
        generation_time = time.time() - start_time
        
        # Decode generated text
        generated_ids = input_ids[0].tolist()
        full_text = self.tokenizer.decode(generated_ids)
        generated_text = self.tokenizer.decode(generated_ids[original_length:])
        
        # Calculate statistics
        avg_token_prob = sum(token_probs) / len(token_probs) if token_probs else 0.0
        tokens_per_second = len(token_probs) / generation_time if generation_time > 0 else 0.0
        
        result = {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': full_text,
            'num_tokens_generated': len(token_probs),
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'average_token_probability': avg_token_prob,
            'settings': {
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
                'repetition_penalty': repetition_penalty,
                'max_new_tokens': max_new_tokens
            }
        }
        
        if show_routing:
            result['routing_stats'] = routing_stats
            if routing_stats:
                avg_recursion = sum(s['avg_recursion_steps'] for s in routing_stats) / len(routing_stats)
                result['average_recursion_steps'] = avg_recursion
                print(f"Average recursion steps: {avg_recursion:.2f}")
        
        return result
    
    def interactive_chat(self):
        """Interactive chat interface"""
        print("MoR Model Interactive Chat")
        print("Commands:")
        print("  'exit' or 'quit' - Exit")
        print("  'presets' - Show generation presets")
        print("  'stats' - Toggle routing statistics")
        print("  'clear' - Clear GPU memory")
        
        show_stats = False
        current_preset = 'balanced'
        
        while True:
            try:
                # Get user input
                prompt = input(f"\n[{current_preset}] Prompt: ").strip()
                
                # Handle commands
                if prompt.lower() in {'exit', 'quit', 'q'}:
                    print("Goodbye!")
                    break
                
                elif prompt.lower() == 'presets':
                    print("\nAvailable presets:")
                    for name, settings in self.generation_presets.items():
                        print(f"  {name}: {settings}")
                    continue
                
                elif prompt.lower() == 'stats':
                    show_stats = not show_stats
                    print(f"Routing statistics: {'ON' if show_stats else 'OFF'}")
                    continue
                
                elif prompt.lower() == 'clear':
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print("GPU memory cleared")
                    continue
                
                elif prompt.lower().startswith('preset:'):
                    preset_name = prompt.split(':', 1)[1].strip()
                    if preset_name in self.generation_presets:
                        current_preset = preset_name
                        print(f"Switched to {preset_name} preset")
                    else:
                        print(f"Unknown preset: {preset_name}")
                    continue
                
                if not prompt:
                    continue
                
                # Get generation settings
                settings = self.generation_presets[current_preset].copy()
                
                # Optional parameter overrides
                max_tokens = input("Max tokens (default 60): ").strip()
                if max_tokens.isdigit():
                    max_tokens = int(max_tokens)
                else:
                    max_tokens = 60
                
                # Generate response
                print(f"\nGenerating with {current_preset} preset...")
                start_time = time.time()
                
                result = self.generate_text(
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    show_routing=show_stats,
                    **settings
                )
                
                # Display result
                print(f"\nGenerated text:\n{result['generated_text']}")
                
                # Show statistics
                print(f"\nStats:")
                print(f"   Time: {result['generation_time']:.2f}s")
                print(f"   Speed: {result['tokens_per_second']:.1f} tokens/s")
                print(f"   Avg prob: {result['average_token_probability']:.3f}")
                
                if show_stats and 'average_recursion_steps' in result:
                    print(f"   Avg recursion: {result['average_recursion_steps']:.2f}")
                
            except KeyboardInterrupt:
                print("\nGeneration interrupted")
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def benchmark(self, num_runs=5):
        """Benchmark the model performance"""
        print(f"Running benchmark ({num_runs} runs)...")
        
        test_prompts = [
            "Hello, how are you?",
            "What is the capital of France?",
            "Explain artificial intelligence.",
            "Write a short story about",
            "The benefits of renewable energy"
        ]
        
        results = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"Testing prompt {i+1}: '{prompt[:30]}...'")
            
            run_results = []
            for run in range(num_runs):
                result = self.generate_text(
                    prompt=prompt,
                    max_new_tokens=50,
                    show_progress=False,
                    **self.generation_presets['balanced']
                )
                run_results.append({
                    'time': result['generation_time'],
                    'tokens_per_second': result['tokens_per_second'],
                    'avg_prob': result['average_token_probability']
                })
            
            # Calculate averages
            avg_time = sum(r['time'] for r in run_results) / num_runs
            avg_speed = sum(r['tokens_per_second'] for r in run_results) / num_runs
            avg_prob = sum(r['avg_prob'] for r in run_results) / num_runs
            
            results.append({
                'prompt': prompt,
                'avg_time': avg_time,
                'avg_speed': avg_speed,
                'avg_probability': avg_prob
            })
            
            print(f"   Avg time: {avg_time:.2f}s")
            print(f"   Avg speed: {avg_speed:.1f} tok/s")
        
        # Overall statistics
        overall_speed = sum(r['avg_speed'] for r in results) / len(results)
        overall_prob = sum(r['avg_probability'] for r in results) / len(results)
        
        print(f"\nBenchmark Results:")
        print(f"   Overall avg speed: {overall_speed:.1f} tokens/sec")
        print(f"   Overall avg probability: {overall_prob:.3f}")
        
        return results

# ====== MAIN INTERFACE ======
def main():
    """Main function with menu interface"""
    try:
        tester = MoRModelTester()
        
        print("\nMoR Model Tester Ready!")
        print("Choose an option:")
        print("1. Interactive chat")
        print("2. Single generation")
        print("3. Benchmark")
        print("4. Exit")
        
        while True:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                tester.interactive_chat()
                break
            
            elif choice == '2':
                prompt = input("Enter prompt: ").strip()
                if prompt:
                    result = tester.generate_text(prompt, show_routing=True)
                    print(f"\nGenerated: {result['generated_text']}")
                    print(f"Time: {result['generation_time']:.2f}s")
                    print(f"Speed: {result['tokens_per_second']:.1f} tokens/s")
                break
            
            elif choice == '3':
                tester.benchmark()
                break
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-4.")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()