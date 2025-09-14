# Backend/model/tokenizer.py — Enhanced MOR Tokenizer v3

import json
import logging
import re
import csv
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Iterator, Set
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np
from functools import lru_cache
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration class for tokenizer settings."""
    vocab_size: int = 32000
    min_freq: int = 2
    max_token_length: int = 50
    enable_subword: bool = True
    subword_merge_ops: int = 5000
    enable_bpe: bool = False
    parallel_workers: int = 4
    chunk_size: int = 1000
    cache_size: int = 10000
    enable_domain_adaptation: bool = True
    domain_vocab_ratio: float = 0.1


class EnhancedMORTokenizer:
    """
    Advanced tokenizer optimized for MOR (Mixture-of-Experts) models with:
    - Subword tokenization (BPE-like)
    - Domain-specific vocabulary adaptation
    - Efficient caching and parallel processing
    - Advanced text preprocessing
    - Robust error handling and recovery
    - Memory-efficient streaming
    """

    def __init__(
        self,
        data_path: Union[str, Path] = None,
        dataset_path: Union[str, Path] = None,
        config: Optional[TokenizerConfig] = None,
        special_tokens: Optional[Dict[str, str]] = None,
        auto_detect_format: bool = True,
        **kwargs
    ):
        # Config setup
        self.config = config or TokenizerConfig(**kwargs)
        
        # Path handling with backward compatibility
        if dataset_path is not None and data_path is None:
            data_path = dataset_path
        elif data_path is None and dataset_path is None:
            data_path = "Backend/data/train.csv"
        
        self.data_path = Path(data_path)
        self.auto_detect_format = auto_detect_format
        
        # Enhanced special tokens for MOR
        self.special_tokens = special_tokens or {
            'pad': '<PAD>',
            'unk': '<UNK>',
            'eos': '<EOS>',
            'bos': '<BOS>',
            'sep': '<SEP>',
            'cls': '<CLS>',
            'mask': '<MASK>',
            'expert_sep': '<EXPERT_SEP>',  # For MOR routing
            'domain_start': '<DOM>',       # Domain markers
            'domain_end': '</DOM>',
            'continue': '<CONTINUE>',      # Continuation token
            'newline': '<NL>'             # Explicit newlines
        }
        
        # Core vocabularies
        self.vocab: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.token_counts = Counter()
        self.subword_vocab: Dict[str, int] = {}
        self.merge_rules: List[Tuple[str, str]] = []
        
        # Domain adaptation
        self.domain_vocabularies: Dict[str, Counter] = defaultdict(Counter)
        self.detected_domains: Set[str] = set()
        
        # Format detection
        self.detected_format = None
        self.format_mapping = {}
        
        # Caching
        self._token_cache = {}
        self._encode_cache = {}
        self._decode_cache = {}
        self._cache_lock = threading.RLock()
        
        # Compiled patterns
        self._token_patterns = {
            'word': re.compile(r'\b\w+\b'),
            'punct': re.compile(r'[^\w\s]'),
            'whitespace': re.compile(r'\s+'),
            'number': re.compile(r'\d+(?:\.\d+)?'),
            'url': re.compile(r'https?://[^\s]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'hashtag': re.compile(r'#\w+'),
            'mention': re.compile(r'@\w+'),
            'code': re.compile(r'`[^`]*`|```[^```]*```'),
            'subword_boundary': re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])')
        }
        
        # Initialize tokenizer
        try:
            self._initialize_tokenizer()
            logger.info(f"✅ Enhanced MOR Tokenizer initialized with {len(self.vocab)} tokens")
        except Exception as e:
            logger.error(f"Tokenizer initialization failed: {e}")
            self._create_emergency_vocab()
    
    def _initialize_tokenizer(self):
        """Initialize the tokenizer with vocabulary building."""
        if self.data_path.exists():
            self.build_vocab()
        else:
            logger.warning(f"Data file not found: {self.data_path}, using minimal vocab")
            self._create_minimal_vocab()
        
        # Build subword vocabulary if enabled
        if self.config.enable_subword:
            self._build_subword_vocab()
        
        # Set up token ID shortcuts
        self._setup_special_token_ids()
    
    def _setup_special_token_ids(self):
        """Set up quick access to special token IDs."""
        self.pad_token_id = self.vocab.get(self.special_tokens['pad'], 0)
        self.unk_token_id = self.vocab.get(self.special_tokens['unk'], 1)
        self.bos_token_id = self.vocab.get(self.special_tokens['bos'], 2)
        self.eos_token_id = self.vocab.get(self.special_tokens['eos'], 3)
        self.sep_token_id = self.vocab.get(self.special_tokens['sep'], 4)
        self.cls_token_id = self.vocab.get(self.special_tokens['cls'], 5)
        self.mask_token_id = self.vocab.get(self.special_tokens['mask'], 6)
        self.expert_sep_token_id = self.vocab.get(self.special_tokens['expert_sep'], 7)
        self.domain_start_token_id = self.vocab.get(self.special_tokens['domain_start'], 8)
        self.domain_end_token_id = self.vocab.get(self.special_tokens['domain_end'], 9)
        self.continue_token_id = self.vocab.get(self.special_tokens['continue'], 10)
        self.newline_token_id = self.vocab.get(self.special_tokens['newline'], 11)
    
    # ===============================
    # VOCABULARY BUILDING
    # ===============================
    
    def _stream_data_chunks(self, chunk_size: int = None) -> Iterator[List[Dict[str, Any]]]:
        """Stream data in chunks for memory efficiency."""
        if not self.data_path.exists():
            return iter([])
        
        chunk_size = chunk_size or self.config.chunk_size
        chunk = []
        
        with open(self.data_path, "r", encoding="utf-8", newline="") as f:
            if self.data_path.suffix == ".csv":
                reader = csv.DictReader(f)
                for row in reader:
                    chunk.append(row)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            elif self.data_path.suffix == ".jsonl":
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        chunk.append(data)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    except json.JSONDecodeError as e:
                        logger.debug(f"Skipping malformed JSON at line {line_num}: {e}")
            elif self.data_path.suffix == ".json":
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i in range(0, len(data), chunk_size):
                            yield data[i:i + chunk_size]
                except Exception as e:
                    logger.error(f"Error parsing JSON: {e}")
        
        if chunk:
            yield chunk
    
    def build_vocab(self):
        """Build vocabulary with advanced features and parallel processing."""
        logger.info("🔨 Building enhanced vocabulary...")
        
        total_samples = 0
        chunk_count = 0
        
        # Process data in chunks
        for chunk in self._stream_data_chunks():
            if not chunk:
                continue
            
            chunk_count += 1
            chunk_size = len(chunk)
            total_samples += chunk_size
            
            # Detect format from first chunk
            if chunk_count == 1 and self.auto_detect_format:
                try:
                    self.detected_format, self.format_mapping = self.detect_dataset_format(chunk)
                    logger.info(f"📂 Detected format: {self.detected_format}")
                except Exception as e:
                    logger.warning(f"Format detection failed: {e}")
                    self.detected_format = "fallback"
                    self.format_mapping = {}
            
            # Process chunk with parallel workers
            if self.config.parallel_workers > 1:
                self._process_chunk_parallel(chunk)
            else:
                self._process_chunk_sequential(chunk)
            
            if chunk_count % 10 == 0:
                logger.info(f"Processed {total_samples} samples in {chunk_count} chunks...")
        
        logger.info(f"📊 Total samples processed: {total_samples}")
        
        # Build final vocabulary
        if not self.token_counts:
            logger.warning("No tokens found, using minimal vocabulary")
            self._create_minimal_vocab()
            return
        
        self._build_vocab_from_counts()
        
        # Domain adaptation
        if self.config.enable_domain_adaptation:
            self._adapt_vocabulary_for_domains()
    
    def _process_chunk_parallel(self, chunk: List[Dict[str, Any]]):
        """Process a chunk with parallel workers."""
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = [
                executor.submit(self._extract_and_analyze_sample, sample) 
                for sample in chunk
            ]
            
            for future in as_completed(futures):
                try:
                    tokens, domain = future.result()
                    self.token_counts.update(tokens)
                    if domain:
                        self.detected_domains.add(domain)
                        self.domain_vocabularies[domain].update(tokens)
                except Exception as e:
                    logger.debug(f"Error processing sample: {e}")
    
    def _process_chunk_sequential(self, chunk: List[Dict[str, Any]]):
        """Process a chunk sequentially."""
        for sample in chunk:
            try:
                tokens, domain = self._extract_and_analyze_sample(sample)
                self.token_counts.update(tokens)
                if domain:
                    self.detected_domains.add(domain)
                    self.domain_vocabularies[domain].update(tokens)
            except Exception as e:
                logger.debug(f"Error processing sample: {e}")
    
    def _extract_and_analyze_sample(self, sample: Dict[str, Any]) -> Tuple[List[str], Optional[str]]:
        """Extract text and analyze domain from a sample."""
        texts = self.extract_text_from_sample(sample)
        domain = self._detect_sample_domain(sample)
        
        all_tokens = []
        for text in texts:
            if text and text.strip():
                tokens = self.advanced_tokenize(text)
                all_tokens.extend(tokens)
        
        return all_tokens, domain
    
    def _detect_sample_domain(self, sample: Dict[str, Any]) -> Optional[str]:
        """Detect domain/category of a sample for domain adaptation."""
        # Look for common domain indicators
        domain_keys = ['domain', 'category', 'topic', 'type', 'label', 'genre', 'subject']
        
        for key in domain_keys:
            if key in sample and sample[key]:
                return str(sample[key]).lower().strip()
        
        # Try to infer from content
        texts = self.extract_text_from_sample(sample)
        if texts:
            content = ' '.join(texts).lower()
            
            # Simple domain detection patterns
            if any(term in content for term in ['code', 'function', 'class', 'import']):
                return 'programming'
            elif any(term in content for term in ['research', 'study', 'analysis', 'findings']):
                return 'academic'
            elif any(term in content for term in ['patient', 'treatment', 'medical', 'health']):
                return 'medical'
            elif any(term in content for term in ['legal', 'court', 'law', 'contract']):
                return 'legal'
        
        return None
    
    def _build_vocab_from_counts(self):
        """Build vocabulary from token counts with smart frequency thresholding."""
        logger.info("📚 Building final vocabulary...")
        
        vocab_id = 0
        
        # Add special tokens first
        for token in self.special_tokens.values():
            if token not in self.vocab:
                self.vocab[token] = vocab_id
                self.id2token[vocab_id] = token
                vocab_id += 1
        
        # Calculate dynamic frequency threshold
        total_tokens = sum(self.token_counts.values())
        frequency_threshold = max(
            self.config.min_freq,
            total_tokens // (self.config.vocab_size * 100)  # Dynamic threshold
        )
        
        # Add tokens by frequency, respecting vocabulary size limit
        remaining_vocab_size = self.config.vocab_size - len(self.vocab)
        
        for token, count in self.token_counts.most_common():
            if len(self.vocab) >= self.config.vocab_size:
                break
            
            if (count >= frequency_threshold and 
                token not in self.vocab and 
                len(token) <= self.config.max_token_length and
                not token.isspace()):
                
                self.vocab[token] = vocab_id
                self.id2token[vocab_id] = token
                vocab_id += 1
        
        logger.info(f"✅ Built vocabulary with {len(self.vocab)} tokens")
        logger.info(f"📈 Token frequency threshold: {frequency_threshold}")
    
    def _adapt_vocabulary_for_domains(self):
        """Adapt vocabulary for detected domains."""
        if not self.detected_domains:
            return
        
        logger.info(f"🎯 Adapting vocabulary for domains: {list(self.detected_domains)}")
        
        domain_vocab_size = int(self.config.vocab_size * self.config.domain_vocab_ratio)
        current_vocab_size = len(self.vocab)
        
        if current_vocab_size >= self.config.vocab_size:
            return
        
        remaining_slots = min(
            domain_vocab_size,
            self.config.vocab_size - current_vocab_size
        )
        
        # Add domain-specific high-frequency tokens
        domain_tokens_added = 0
        vocab_id = max(self.vocab.values()) + 1
        
        for domain in self.detected_domains:
            if domain_tokens_added >= remaining_slots:
                break
            
            domain_counter = self.domain_vocabularies[domain]
            
            for token, count in domain_counter.most_common():
                if (domain_tokens_added >= remaining_slots or 
                    len(self.vocab) >= self.config.vocab_size):
                    break
                
                if (token not in self.vocab and 
                    count >= self.config.min_freq * 2 and  # Higher threshold for domain tokens
                    len(token) <= self.config.max_token_length):
                    
                    self.vocab[token] = vocab_id
                    self.id2token[vocab_id] = token
                    vocab_id += 1
                    domain_tokens_added += 1
        
        logger.info(f"🎯 Added {domain_tokens_added} domain-specific tokens")
    
    # ===============================
    # SUBWORD TOKENIZATION
    # ===============================
    
    def _build_subword_vocab(self):
        """Build subword vocabulary using BPE-like approach."""
        if not self.config.enable_bpe:
            return
        
        logger.info("🔤 Building subword vocabulary...")
        
        # Get character-level statistics
        char_pairs = Counter()
        word_freqs = {}
        
        # Collect word frequencies and character pairs
        for word, freq in self.token_counts.most_common(10000):  # Limit for efficiency
            if len(word) < 2:
                continue
            
            word_chars = list(word)
            word_freqs[word] = freq
            
            for i in range(len(word_chars) - 1):
                pair = (word_chars[i], word_chars[i + 1])
                char_pairs[pair] += freq
        
        # Learn BPE merges
        self.merge_rules = []
        
        for _ in range(self.config.subword_merge_ops):
            if not char_pairs:
                break
            
            # Find most frequent pair
            best_pair = char_pairs.most_common(1)[0][0]
            
            # Apply merge
            new_word_freqs = {}
            new_char_pairs = Counter()
            
            for word, freq in word_freqs.items():
                new_word = word
                if f"{best_pair[0]}{best_pair[1]}" in word:
                    new_word = word.replace(
                        f"{best_pair[0]}{best_pair[1]}", 
                        f"{best_pair[0]}@@{best_pair[1]}"
                    )
                
                new_word_freqs[new_word] = freq
                
                # Update pair counts for new word
                chars = new_word.split('@@')
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    new_char_pairs[pair] += freq
            
            # Update for next iteration
            word_freqs = new_word_freqs
            char_pairs = new_char_pairs
            self.merge_rules.append(best_pair)
        
        logger.info(f"🔤 Learned {len(self.merge_rules)} BPE merge rules")
    
    # ===============================
    # TOKENIZATION
    # ===============================
    
    @lru_cache(maxsize=10000)
    def advanced_tokenize(self, text: str) -> List[str]:
        """Advanced tokenization with multiple strategies."""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # Check cache first
        text_hash = hashlib.md5(text.encode()).hexdigest()
        with self._cache_lock:
            if text_hash in self._token_cache:
                return self._token_cache[text_hash]
        
        tokens = []
        
        # Handle special patterns first
        remaining_text = text
        
        # URLs
        for match in self._token_patterns['url'].finditer(text):
            tokens.append(f"<URL>{match.group()}</URL>")
            remaining_text = remaining_text.replace(match.group(), ' ')
        
        # Email addresses
        for match in self._token_patterns['email'].finditer(remaining_text):
            tokens.append(f"<EMAIL>{match.group()}</EMAIL>")
            remaining_text = remaining_text.replace(match.group(), ' ')
        
        # Code blocks
        for match in self._token_patterns['code'].finditer(remaining_text):
            code_tokens = self._tokenize_code(match.group())
            tokens.extend(code_tokens)
            remaining_text = remaining_text.replace(match.group(), ' ')
        
        # Hashtags and mentions
        for match in self._token_patterns['hashtag'].finditer(remaining_text):
            tokens.append(match.group())
            remaining_text = remaining_text.replace(match.group(), ' ')
        
        for match in self._token_patterns['mention'].finditer(remaining_text):
            tokens.append(match.group())
            remaining_text = remaining_text.replace(match.group(), ' ')
        
        # Regular tokenization for remaining text
        remaining_text = remaining_text.lower()
        
        # Split by whitespace and punctuation
        parts = self._token_patterns['whitespace'].split(remaining_text)
        
        for part in parts:
            if not part.strip():
                continue
            
            # Handle punctuation
            punct_tokens = self._tokenize_with_punctuation(part)
            tokens.extend(punct_tokens)
        
        # Apply subword tokenization if enabled
        if self.config.enable_subword and self.merge_rules:
            tokens = self._apply_subword_tokenization(tokens)
        
        # Cache result
        with self._cache_lock:
            if len(self._token_cache) < self.config.cache_size:
                self._token_cache[text_hash] = tokens
        
        return tokens
    
    def _tokenize_with_punctuation(self, text: str) -> List[str]:
        """Tokenize text handling punctuation separately."""
        if not text:
            return []
        
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isalnum() or char in "_-":
                current_token += char
            else:
                if current_token:
                    # Apply camelCase splitting if enabled
                    if self.config.enable_subword:
                        subtokens = self._split_camel_case(current_token)
                        tokens.extend(subtokens)
                    else:
                        tokens.append(current_token)
                    current_token = ""
                
                if not char.isspace():
                    tokens.append(char)
        
        if current_token:
            if self.config.enable_subword:
                subtokens = self._split_camel_case(current_token)
                tokens.extend(subtokens)
            else:
                tokens.append(current_token)
        
        return tokens
    
    def _split_camel_case(self, token: str) -> List[str]:
        """Split camelCase tokens into subtokens."""
        if len(token) <= 2:
            return [token]
        
        splits = self._token_patterns['subword_boundary'].split(token)
        return [s for s in splits if s]
    
    def _tokenize_code(self, code_text: str) -> List[str]:
        """Special tokenization for code blocks."""
        # Remove code block markers
        code_text = code_text.strip('`')
        
        tokens = ['<CODE_START>']
        
        # Simple code tokenization
        code_tokens = re.findall(r'\w+|[^\w\s]', code_text)
        tokens.extend(code_tokens)
        
        tokens.append('<CODE_END>')
        return tokens
    
    def _apply_subword_tokenization(self, tokens: List[str]) -> List[str]:
        """Apply learned BPE rules to tokens."""
        if not self.merge_rules:
            return tokens
        
        result_tokens = []
        
        for token in tokens:
            if len(token) <= 2 or not token.isalpha():
                result_tokens.append(token)
                continue
            
            # Apply BPE rules
            chars = list(token)
            
            for merge_pair in self.merge_rules:
                merged_chars = []
                i = 0
                
                while i < len(chars):
                    if (i < len(chars) - 1 and 
                        chars[i] == merge_pair[0] and 
                        chars[i + 1] == merge_pair[1]):
                        merged_chars.append(f"{chars[i]}@@{chars[i + 1]}")
                        i += 2
                    else:
                        merged_chars.append(chars[i])
                        i += 1
                
                chars = merged_chars
            
            # Convert back to tokens
            subword_tokens = []
            for char_group in chars:
                if '@@' in char_group:
                    subword_tokens.extend(char_group.split('@@'))
                else:
                    subword_tokens.append(char_group)
            
            result_tokens.extend(subword_tokens)
        
        return result_tokens
    
    # ===============================
    # ENCODING/DECODING
    # ===============================
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = False,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """Enhanced encoding with advanced options."""
        if not text:
            return []
        
        # Check cache
        cache_key = f"{text}_{add_special_tokens}_{max_length}_{padding}_{truncation}"
        with self._cache_lock:
            if cache_key in self._encode_cache:
                return self._encode_cache[cache_key]
        
        tokens = self.advanced_tokenize(text)
        token_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        # Handle max length
        if max_length:
            if truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
                if add_special_tokens and token_ids:
                    token_ids[-1] = self.eos_token_id
            
            if padding and len(token_ids) < max_length:
                token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        # Cache result
        with self._cache_lock:
            if len(self._encode_cache) < self.config.cache_size:
                self._encode_cache[cache_key] = token_ids
        
        return token_ids
    
    def decode(
        self, 
        token_ids: List[int], 
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """Enhanced decoding with cleanup options."""
        if not token_ids:
            return ""
        
        # Check cache
        cache_key = f"{hash(tuple(token_ids))}_{skip_special_tokens}_{clean_up_tokenization_spaces}"
        with self._cache_lock:
            if cache_key in self._decode_cache:
                return self._decode_cache[cache_key]
        
        tokens = []
        special_token_values = set(self.special_tokens.values()) if skip_special_tokens else set()
        
        for token_id in token_ids:
            if token_id in self.id2token:
                token = self.id2token[token_id]
                if not skip_special_tokens or token not in special_token_values:
                    tokens.append(token)
        
        # Join tokens
        text = " ".join(tokens)
        
        if clean_up_tokenization_spaces:
            # Clean up tokenization artifacts
            text = re.sub(r' +', ' ', text)  # Multiple spaces
            text = re.sub(r' ([.!?,:;])', r'\1', text)  # Punctuation
            text = re.sub(r'( |^)@ ', r'\1@', text)  # Mentions
            text = re.sub(r'( |^)# ', r'\1#', text)  # Hashtags
            text = text.strip()
        
        # Handle subword tokens
        if self.config.enable_subword:
            text = self._merge_subword_tokens(text)
        
        # Cache result
        with self._cache_lock:
            if len(self._decode_cache) < self.config.cache_size:
                self._decode_cache[cache_key] = text
        
        return text
    
    def _merge_subword_tokens(self, text: str) -> str:
        """Merge subword tokens back into words."""
        # Handle BPE-style merging
        text = re.sub(r'@@', '', text)
        
        # Handle code blocks
        text = re.sub(r'<CODE_START>(.*?)<CODE_END>', r'`\1`', text)
        
        # Handle URLs and emails
        text = re.sub(r'<URL>(.*?)</URL>', r'\1', text)
        text = re.sub(r'<EMAIL>(.*?)</EMAIL>', r'\1', text)
        
        return text
    
    def encode_pair(
        self,
        text1: str,
        text2: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """Encode text pair for tasks like classification or similarity."""
        tokens1 = self.encode(text1, add_special_tokens=False)
        tokens2 = self.encode(text2, add_special_tokens=False)
        
        if add_special_tokens:
            token_ids = (
                [self.bos_token_id] + 
                tokens1 + 
                [self.sep_token_id] + 
                tokens2 + 
                [self.eos_token_id]
            )
        else:
            token_ids = tokens1 + [self.sep_token_id] + tokens2
        
        # Handle max length and padding
        if max_length:
            if truncation and len(token_ids) > max_length:
                # Truncate second text first
                available_length = max_length - len(tokens1) - 3  # BOS, SEP, EOS
                if available_length > 0:
                    tokens2 = tokens2[:available_length]
                    token_ids = (
                        [self.bos_token_id] + 
                        tokens1 + 
                        [self.sep_token_id] + 
                        tokens2 + 
                        [self.eos_token_id]
                    )
                else:
                    token_ids = token_ids[:max_length]
                    if add_special_tokens and token_ids:
                        token_ids[-1] = self.eos_token_id
            
            if padding and len(token_ids) < max_length:
                token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        return token_ids
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[List[int]]]:
        """Batch encode multiple texts efficiently."""
        if self.config.parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = [
                    executor.submit(
                        self.encode, text, add_special_tokens, max_length, padding, truncation
                    ) for text in texts
                ]
                token_ids_list = [future.result() for future in futures]
        else:
            token_ids_list = [
                self.encode(text, add_special_tokens, max_length, padding, truncation)
                for text in texts
            ]
        
        # Create attention masks
        attention_masks = []
        for token_ids in token_ids_list:
            mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]
            attention_masks.append(mask)
        
        return {
            'input_ids': token_ids_list,
            'attention_mask': attention_masks
        }
    
    def batch_decode(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> List[str]:
        """Batch decode multiple token sequences."""
        if self.config.parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = [
                    executor.submit(
                        self.decode, token_ids, skip_special_tokens, clean_up_tokenization_spaces
                    ) for token_ids in token_ids_list
                ]
                texts = [future.result() for future in futures]
        else:
            texts = [
                self.decode(token_ids, skip_special_tokens, clean_up_tokenization_spaces)
                for token_ids in token_ids_list
            ]
        
        return texts
    
    # ===============================
    # MOR-SPECIFIC METHODS
    # ===============================
    
    def encode_for_mor(
        self,
        text: str,
        expert_routing_info: Optional[Dict] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Encode text specifically for MOR model with expert routing information."""
        # Regular encoding
        token_ids = self.encode(text, add_special_tokens=True)
        
        # Add domain markers if specified
        if domain:
            domain_start = [self.domain_start_token_id]
            domain_marker = self.encode(domain, add_special_tokens=False)
            domain_end = [self.domain_end_token_id]
            
            token_ids = domain_start + domain_marker + domain_end + token_ids
        
        # Create expert routing tokens if provided
        expert_tokens = []
        if expert_routing_info:
            for expert_id in expert_routing_info.get('preferred_experts', []):
                expert_tokens.extend([self.expert_sep_token_id, expert_id])
        
        return {
            'input_ids': token_ids,
            'expert_routing_tokens': expert_tokens,
            'domain': domain,
            'attention_mask': [1] * len(token_ids)
        }
    
    def create_expert_routing_sequence(self, expert_ids: List[int]) -> List[int]:
        """Create a sequence of expert routing tokens."""
        routing_sequence = []
        for expert_id in expert_ids:
            routing_sequence.extend([self.expert_sep_token_id, expert_id])
        return routing_sequence
    
    def extract_expert_routing_from_tokens(self, token_ids: List[int]) -> List[int]:
        """Extract expert routing information from token sequence."""
        expert_ids = []
        i = 0
        while i < len(token_ids) - 1:
            if token_ids[i] == self.expert_sep_token_id:
                expert_ids.append(token_ids[i + 1])
                i += 2
            else:
                i += 1
        return expert_ids
    
    # ===============================
    # FORMAT DETECTION & TEXT EXTRACTION
    # ===============================
    
    def detect_dataset_format(self, sample_data: List[Dict[str, Any]]) -> Tuple[str, Dict[str, str]]:
        """Enhanced format detection with more robust patterns."""
        if not sample_data:
            return "empty", {}
        
        sample = sample_data[0]
        
        # Common format patterns
        format_patterns = {
            'instruction_following': ['instruction', 'input', 'output', 'response'],
            'qa_pairs': ['question', 'answer', 'context'],
            'conversation': ['human', 'assistant', 'user', 'bot'],
            'classification': ['text', 'label', 'category', 'class'],
            'completion': ['prompt', 'completion', 'text'],
            'translation': ['source', 'target', 'src', 'tgt'],
            'summarization': ['document', 'summary', 'article', 'abstract'],
            'code': ['code', 'function', 'implementation', 'solution'],
            'medical': ['patient', 'diagnosis', 'treatment', 'symptoms'],
            'legal': ['case', 'ruling', 'law', 'contract'],
        }
        
        # Score each format
        format_scores = {}
        sample_keys = set(str(k).lower() for k in sample.keys())
        
        for format_name, keywords in format_patterns.items():
            score = sum(1 for kw in keywords if kw in sample_keys)
            if score > 0:
                format_scores[format_name] = score
        
        if not format_scores:
            return "generic", self._create_generic_mapping(sample)
        
        # Get best matching format
        best_format = max(format_scores.items(), key=lambda x: x[1])[0]
        mapping = self._create_format_mapping(best_format, sample)
        
        return best_format, mapping
    
    def _create_format_mapping(self, format_type: str, sample: Dict[str, Any]) -> Dict[str, str]:
        """Create field mapping for detected format."""
        mappings = {
            'instruction_following': {
                'instruction': self._find_key(sample, ['instruction', 'task', 'prompt']),
                'input': self._find_key(sample, ['input', 'context', 'given']),
                'output': self._find_key(sample, ['output', 'response', 'answer', 'completion'])
            },
            'qa_pairs': {
                'question': self._find_key(sample, ['question', 'query', 'q']),
                'answer': self._find_key(sample, ['answer', 'response', 'a']),
                'context': self._find_key(sample, ['context', 'passage', 'document'])
            },
            'conversation': {
                'human': self._find_key(sample, ['human', 'user', 'person']),
                'assistant': self._find_key(sample, ['assistant', 'bot', 'ai', 'system'])
            },
            'classification': {
                'text': self._find_key(sample, ['text', 'content', 'document']),
                'label': self._find_key(sample, ['label', 'class', 'category'])
            },
            'completion': {
                'prompt': self._find_key(sample, ['prompt', 'input', 'prefix']),
                'completion': self._find_key(sample, ['completion', 'output', 'suffix'])
            }
        }
        
        return mappings.get(format_type, self._create_generic_mapping(sample))
    
    def _create_generic_mapping(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Create generic mapping for unknown formats."""
        text_fields = []
        for key, value in sample.items():
            if isinstance(value, str) and len(value.strip()) > 10:
                text_fields.append(key)
        
        return {'text_fields': text_fields}
    
    def _find_key(self, sample: Dict[str, Any], candidates: List[str]) -> Optional[str]:
        """Find the first matching key from candidates in sample."""
        sample_keys = {str(k).lower(): str(k) for k in sample.keys()}
        
        for candidate in candidates:
            if candidate.lower() in sample_keys:
                return sample_keys[candidate.lower()]
        
        return None
    
    def extract_text_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        """Enhanced text extraction using format mapping."""
        texts = []
        
        if not self.format_mapping:
            # Fallback: extract all string values
            for value in sample.values():
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
            return texts
        
        # Use format-specific extraction
        if self.detected_format == 'instruction_following':
            for field in ['instruction', 'input', 'output']:
                key = self.format_mapping.get(field)
                if key and key in sample:
                    text = self._safe_extract_text(sample[key])
                    if text:
                        texts.append(text)
        
        elif self.detected_format == 'qa_pairs':
            for field in ['question', 'answer', 'context']:
                key = self.format_mapping.get(field)
                if key and key in sample:
                    text = self._safe_extract_text(sample[key])
                    if text:
                        texts.append(text)
        
        elif self.detected_format == 'conversation':
            for field in ['human', 'assistant']:
                key = self.format_mapping.get(field)
                if key and key in sample:
                    text = self._safe_extract_text(sample[key])
                    if text:
                        texts.append(text)
        
        else:
            # Generic extraction
            text_fields = self.format_mapping.get('text_fields', [])
            for field in text_fields:
                if field in sample:
                    text = self._safe_extract_text(sample[field])
                    if text:
                        texts.append(text)
        
        # Fallback if no texts found
        if not texts:
            for value in sample.values():
                if isinstance(value, str) and value.strip():
                    texts.append(value.strip())
        
        return texts
    
    def _safe_extract_text(self, value: Any) -> Optional[str]:
        """Safely extract text from various value types."""
        if isinstance(value, str):
            return value.strip() if value.strip() else None
        elif isinstance(value, (list, tuple)) and value:
            # Handle list of texts
            texts = []
            for item in value:
                if isinstance(item, str) and item.strip():
                    texts.append(item.strip())
            return ' '.join(texts) if texts else None
        elif isinstance(value, dict):
            # Handle nested dictionaries
            texts = []
            for v in value.values():
                text = self._safe_extract_text(v)
                if text:
                    texts.append(text)
            return ' '.join(texts) if texts else None
        else:
            # Convert to string if possible
            try:
                text = str(value).strip()
                return text if text and text != 'None' else None
            except:
                return None
    
    # ===============================
    # VOCABULARY UTILITIES
    # ===============================
    
    def _create_minimal_vocab(self):
        """Create minimal vocabulary with just special tokens."""
        logger.info("Creating minimal vocabulary...")
        self.vocab.clear()
        self.id2token.clear()
        
        vocab_id = 0
        for token in self.special_tokens.values():
            self.vocab[token] = vocab_id
            self.id2token[vocab_id] = token
            vocab_id += 1
        
        logger.info(f"Created minimal vocabulary with {len(self.vocab)} special tokens")
    
    def _create_emergency_vocab(self):
        """Emergency vocabulary creation if all else fails."""
        logger.warning("Creating emergency fallback vocabulary...")
        
        self.vocab = {token: i for i, token in enumerate(self.special_tokens.values())}
        self.id2token = {i: token for token, i in self.vocab.items()}
        
        # Add basic English tokens
        basic_tokens = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        ]
        
        vocab_id = len(self.vocab)
        for token in basic_tokens:
            if token not in self.vocab:
                self.vocab[token] = vocab_id
                self.id2token[vocab_id] = token
                vocab_id += 1
    
    def get_vocab_stats(self) -> Dict[str, Any]:
        """Get comprehensive vocabulary statistics."""
        return {
            'total_tokens': len(self.vocab),
            'special_tokens': len(self.special_tokens),
            'regular_tokens': len(self.vocab) - len(self.special_tokens),
            'subword_rules': len(self.merge_rules) if hasattr(self, 'merge_rules') else 0,
            'detected_domains': list(self.detected_domains),
            'format_info': {
                'detected_format': self.detected_format,
                'format_mapping_keys': list(self.format_mapping.keys()) if self.format_mapping else []
            },
            'cache_stats': {
                'token_cache_size': len(self._token_cache),
                'encode_cache_size': len(self._encode_cache),
                'decode_cache_size': len(self._decode_cache)
            }
        }
    
    def clear_cache(self):
        """Clear all internal caches."""
        with self._cache_lock:
            self._token_cache.clear()
            self._encode_cache.clear()
            self._decode_cache.clear()
        logger.info("Cleared tokenizer caches")
    
    def resize_vocab(self, new_size: int):
        """Resize vocabulary to new size."""
        if new_size <= len(self.special_tokens):
            logger.warning(f"New size {new_size} too small, keeping current size")
            return
        
        if new_size >= len(self.vocab):
            logger.info(f"New size {new_size} larger than current, no resize needed")
            return
        
        # Keep special tokens and most frequent regular tokens
        special_tokens_count = len(self.special_tokens)
        tokens_to_keep = new_size - special_tokens_count
        
        # Get regular tokens sorted by frequency
        regular_tokens = [
            (token, count) for token, count in self.token_counts.most_common()
            if token in self.vocab and token not in self.special_tokens.values()
        ]
        
        # Rebuild vocab
        new_vocab = {}
        new_id2token = {}
        vocab_id = 0
        
        # Add special tokens first
        for token in self.special_tokens.values():
            new_vocab[token] = vocab_id
            new_id2token[vocab_id] = token
            vocab_id += 1
        
        # Add most frequent regular tokens
        for token, _ in regular_tokens[:tokens_to_keep]:
            new_vocab[token] = vocab_id
            new_id2token[vocab_id] = token
            vocab_id += 1
        
        self.vocab = new_vocab
        self.id2token = new_id2token
        self.config.vocab_size = new_size
        
        # Clear caches as they're now invalid
        self.clear_cache()
        
        logger.info(f"Resized vocabulary to {len(self.vocab)} tokens")
    
    # ===============================
    # SAVE/LOAD
    # ===============================
    
    def save(self, save_path: Union[str, Path], save_format: str = 'json'):
        """Save tokenizer with multiple format options."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'vocab': self.vocab,
            'id2token': self.id2token,
            'special_tokens': self.special_tokens,
            'config': {
                'vocab_size': self.config.vocab_size,
                'min_freq': self.config.min_freq,
                'max_token_length': self.config.max_token_length,
                'enable_subword': self.config.enable_subword,
                'subword_merge_ops': self.config.subword_merge_ops,
                'enable_bpe': self.config.enable_bpe,
                'parallel_workers': self.config.parallel_workers,
                'chunk_size': self.config.chunk_size,
                'cache_size': self.config.cache_size,
                'enable_domain_adaptation': self.config.enable_domain_adaptation,
                'domain_vocab_ratio': self.config.domain_vocab_ratio
            },
            'format_info': {
                'detected_format': self.detected_format,
                'format_mapping': self.format_mapping
            },
            'subword_info': {
                'merge_rules': getattr(self, 'merge_rules', []),
                'subword_vocab': getattr(self, 'subword_vocab', {})
            },
            'domain_info': {
                'detected_domains': list(self.detected_domains),
                'domain_vocabularies': {
                    domain: dict(counter.most_common(1000))  # Save top 1000 per domain
                    for domain, counter in self.domain_vocabularies.items()
                }
            },
            'version': '3.0',
            'stats': self.get_vocab_stats()
        }
        
        if save_format.lower() == 'json':
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(tokenizer_data, f, indent=2, ensure_ascii=False)
        elif save_format.lower() == 'pickle':
            with open(save_path, 'wb') as f:
                pickle.dump(tokenizer_data, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        logger.info(f"💾 Enhanced MOR Tokenizer saved to {save_path}")
    
    @classmethod
    def load(
        cls, 
        load_path: Union[str, Path], 
        load_format: str = 'auto'
    ) -> "EnhancedMORTokenizer":
        """Load tokenizer with automatic format detection."""
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {load_path}")
        
        # Auto-detect format
        if load_format == 'auto':
            if load_path.suffix.lower() == '.json':
                load_format = 'json'
            elif load_path.suffix.lower() in ['.pkl', '.pickle']:
                load_format = 'pickle'
            else:
                # Try JSON first
                load_format = 'json'
        
        # Load data
        try:
            if load_format.lower() == 'json':
                with open(load_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif load_format.lower() == 'pickle':
                with open(load_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported load format: {load_format}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Create instance
        obj = cls.__new__(cls)
        
        # Restore basic attributes
        obj.vocab = data['vocab']
        obj.id2token = {int(k): v for k, v in data['id2token'].items()}
        obj.special_tokens = data['special_tokens']
        
        # Restore config
        config_data = data.get('config', {})
        obj.config = TokenizerConfig(**config_data)
        
        # Restore format info
        format_info = data.get('format_info', {})
        obj.detected_format = format_info.get('detected_format')
        obj.format_mapping = format_info.get('format_mapping', {})
        
        # Restore subword info
        subword_info = data.get('subword_info', {})
        obj.merge_rules = subword_info.get('merge_rules', [])
        obj.subword_vocab = subword_info.get('subword_vocab', {})
        
        # Restore domain info
        domain_info = data.get('domain_info', {})
        obj.detected_domains = set(domain_info.get('detected_domains', []))
        obj.domain_vocabularies = defaultdict(Counter)
        for domain, tokens in domain_info.get('domain_vocabularies', {}).items():
            obj.domain_vocabularies[domain] = Counter(tokens)
        
        # Initialize other attributes
        obj.token_counts = Counter()
        obj._token_cache = {}
        obj._encode_cache = {}
        obj._decode_cache = {}
        obj._cache_lock = threading.RLock()
        
        # Compile patterns
        obj._token_patterns = {
            'word': re.compile(r'\b\w+\b'),
            'punct': re.compile(r'[^\w\s]'),
            'whitespace': re.compile(r'\s+'),
            'number': re.compile(r'\d+(?:\.\d+)?'),
            'url': re.compile(r'https?://[^\s]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'hashtag': re.compile(r'#\w+'),
            'mention': re.compile(r'@\w+'),
            'code': re.compile(r'`[^`]*`|```[^```]*```'),
            'subword_boundary': re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|(?<=[0-9])(?=[A-Za-z])')
        }
        
        # Set up special token IDs
        obj._setup_special_token_ids()
        
        logger.info(f"✅ Enhanced MOR Tokenizer loaded from {load_path}")
        logger.info(f"📊 Loaded {len(obj.vocab)} tokens, detected format: {obj.detected_format}")
        
        return obj
    
    def export_vocab(self, export_path: Union[str, Path], format: str = 'txt'):
        """Export vocabulary in various formats."""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'txt':
            with open(export_path, 'w', encoding='utf-8') as f:
                for token_id in sorted(self.id2token.keys()):
                    token = self.id2token[token_id]
                    count = self.token_counts.get(token, 0)
                    f.write(f"{token_id}\t{token}\t{count}\n")
        
        elif format.lower() == 'json':
            vocab_data = {
                'vocab': self.vocab,
                'token_counts': dict(self.token_counts),
                'special_tokens': self.special_tokens,
                'stats': self.get_vocab_stats()
            }
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'csv':
            with open(export_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['token_id', 'token', 'count', 'is_special'])
                
                for token_id in sorted(self.id2token.keys()):
                    token = self.id2token[token_id]
                    count = self.token_counts.get(token, 0)
                    is_special = token in self.special_tokens.values()
                    writer.writerow([token_id, token, count, is_special])
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"📤 Vocabulary exported to {export_path}")
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self.vocab
    
    def __repr__(self) -> str:
        """String representation of the tokenizer."""
        return (
            f"EnhancedMORTokenizer("
            f"vocab_size={len(self.vocab)}, "
            f"format={self.detected_format}, "
            f"domains={len(self.detected_domains)}, "
            f"subword_enabled={self.config.enable_subword})"
        )


# Backward compatibility aliases
OptimizedTokenizer = EnhancedMORTokenizer
Tokenizer = EnhancedMORTokenizer