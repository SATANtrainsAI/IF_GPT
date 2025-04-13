
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision.utils import make_grid, save_image
from pathlib import Path
import argparse
import logging
from contextlib import nullcontext
import time
import sys

# Import custom utilities and models
import wandb

from model import Encoder, GPT, IFModel2


import os
import random
import math
import hashlib
from PIL import Image




# -------------------------
# 1) Utility: infinite loader
# -------------------------

# -------------------------
# 1) Utility: infinite loader
# -------------------------
def cycle(dl):
    """
    Creates an infinite generator from a DataLoader.
    """
    while True:
        for data in dl:
            yield data

# -------------------------
# 2) Custom collate function for zero-padding
# -------------------------
def next_multiple_of(value, multiple):
    """
    Round 'value' up to the next integer multiple of 'multiple'.
    Example: next_multiple_of(61, 16) = 64
    """
    return ((value + multiple - 1) // multiple) * multiple

def collate_with_padding(batch, patch_size, block_size = 264):
    """
    Given a list of (image_tensor, seq_string, protein_id) tuples, finds the largest
    spatial dimension in the batch (across all images), rounds it up to a multiple
    of patch_size, and zero-pads all images accordingly.
    Returns (padded_images, list_of_sequences, list_of_protein_ids).
    """
    # Remove any items that failed to load or were None.
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None

    images = []
    sequences = []
    names = []
    for (img, seq, protein_id) in batch:
        images.append(img)
        sequences.append(seq)
        names.append(protein_id)

    max_h = max(im.shape[1] for im in images)
    max_w = max(im.shape[2] for im in images)
    L = max(max_h, max_w)
    L_pad = next_multiple_of(L, patch_size)

    padded = []
    for im in images:
        c, h, w = im.shape
        pad_h = L_pad - h
        pad_w = L_pad - w
        # F.pad takes (left, right, top, bottom)
        padded_im = F.pad(im, (0, pad_w, 0, pad_h), mode='constant', value=0.)
        padded.append(padded_im)
     
    L_pad = L_pad + 4
    padded_sequences = [
            list(seq)[:L_pad] + [0] * max(0, L_pad - len(seq)) for seq in sequences
        ]
    

    input_ids = torch.tensor(padded_sequences, dtype=torch.long)
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]
 

    return torch.stack(padded, dim=0), inputs, targets, names

# -------------------------
# 3) Deterministic subset assignment
# -------------------------
def assign_subset(path, valid_frac):
    """
    Returns 'val' if a deterministic hash of the file path is less than valid_frac,
    otherwise 'train'. This ensures each file is always assigned to the same subset.
    """
    h = hashlib.md5(str(path).encode('utf-8')).hexdigest()
    h_int = int(h, 16)
    ratio = h_int / (2**128)
    return 'val' if ratio < valid_frac else 'train'


class IterableImageDatasetUni(IterableDataset):
    """
    Streams images from all bins under a parent folder (excluding a given set)
    without loading all file paths into memory.
    
    Optionally splits images into 'train' and 'val' subsets based on a deterministic hash.

    **Now returns (image_tensor, seq_string, protein_id).**
    """
    def __init__(self, folder,
                 exts=['jpg', 'jpeg', 'png'],
                 normalize_mean=(0.4081, 0.5336, 0.4414),
                 normalize_std=(0.2538, 0.2752, 0.2157),
                 shuffle_buffer_size=1000,
                 subset=None,
                 valid_frac=0.01,
                 skip_count=0):
        super().__init__()
        self.folder = Path(folder)
        self.exts = set(ext.lower() for ext in exts)
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])
        self.shuffle_buffer_size = shuffle_buffer_size
        self.subset = subset
        self.skip_count = skip_count
        self.valid_frac = valid_frac
        # Hardcode the path where FASTAs are stored:
        self.fasta_dir = Path("/scratch/mnaseri1/fasta") 

        # Exclude these bins (if needed):
        #, Path("/scratch/sbayakhm/img/bin_193_256"), Path("/scratch/sbayakhm/img/bin_257_320") = >done,  Path("/scratch/xwang213/img/bin_129_192"), Path("/scratch/xwang213/img/bin_193_256"), Path("/scratch/xwang213/img/bin_257_320"), Path("/scratch/xwang213/img/bin_385_448"), Path("/scratch/xwang213/img/bin_449_512")
        self.yes = {
        Path("/scratch/mnaseri1/img2/bin_129_192"), 
        }

        print(f"Streaming images from {self.folder} (excluding specified bins) for subset: {self.subset} and skipping first {self.skip_count} items.")

    def _iter_paths(self):
        bin_folders = sorted(self.folder.iterdir(), key=lambda p: str(p).lower())
        for bin_folder in bin_folders:
            if bin_folder.is_dir() and bin_folder in self.yes:
                print(f"Now streaming from bin: {bin_folder}")
                for root, _, files in os.walk(bin_folder):
                    for file in files:
                        if file.split('.')[-1].lower() in self.exts:
                            yield Path(root) / file

    def _stream_shuffle(self, iterator):
        buffer = []
        for item in iterator:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                for elem in buffer:
                    yield elem
                buffer = []
        if buffer:
            random.shuffle(buffer)
            yield from buffer

    def __iter__(self):
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        worker_info = torch.utils.data.get_worker_info()

        paths_iter = self._iter_paths()
        paths_iter = self._stream_shuffle(paths_iter)
        if self.subset in ['train', 'val']:
            paths_iter = (p for p in paths_iter if assign_subset(p, self.valid_frac) == self.subset)
        #paths_iter = (p for i, p in enumerate(paths_iter) if i % world_size == rank)
        paths_iter = (p for i, p in enumerate(paths_iter) if i >= self.skip_count and i % world_size == rank)

        if worker_info is not None:
            def worker_shard():
                for i, path in enumerate(paths_iter):
                    if i % worker_info.num_workers == worker_info.id:
                        yield self._load_image(path)
            return worker_shard()
        else:
            return (self._load_image(p) for p in paths_iter)

    def _load_image(self, path):
        """
        Loads the image as before, plus loads the matching FASTA sequence.
        Returns (img_tensor, seq_string, protein_id).
        """
        try:
            img = Image.open(path).convert("RGB")
            img_tensor = self.transform(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

        protein_id = path.stem
        fasta_path = self.fasta_dir / f"{protein_id}.fasta"
        seq_str = ""
        try:
            with open(fasta_path, "r") as f:
                lines = f.read().splitlines()
                if len(lines) > 1:
                    seq_str = lines[1].strip()
        except FileNotFoundError:
            seq_str = ""
        seq_token = tokenizer.encode(seq_str, add_special_tokens=True)

        return (img_tensor, seq_token, protein_id)

    def __getitem__(self, index):
        raise NotImplementedError("IterableImageDatasetUni does not support __getitem__.")



def train_for_all(folder, patch_size, batch_size, ddp, ddp_world_size, ddp_rank, num_workers, valid_frac=0.0):
    """
    For a parent folder, creates IterableImageDatasetUni instances for streaming images from all bins (excluding a given set)
    with optional train/val splitting.
    
    Returns two infinite dataloaders, each yielding (padded_images, list_of_sequences, list_of_protein_ids).
    """
    if valid_frac > 0:
        train_ds = IterableImageDatasetUni(folder=folder, subset='train', valid_frac=valid_frac)
        val_ds = IterableImageDatasetUni(folder=folder, subset='val', valid_frac=valid_frac)
    else:
        train_ds = IterableImageDatasetUni(folder=folder)
        val_ds = None

    train_dl = cycle(DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda batch: collate_with_padding(batch, patch_size=patch_size)
    ))
    if val_ds is not None:
        val_dl = cycle(DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda batch: collate_with_padding(batch, patch_size=patch_size)
        ))
    else:
        val_dl = None

    return train_dl, val_dl


def exists(val):
    return val is not None

def print0(*args, **kwargs):
    """
    Modified print that only prints from the master process.
    If not a distributed run, it behaves like a regular print.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def get_eos_positions(seq_tensor, eos_id):
    """
    seq_tensor: (B, T) of token IDs
    Returns a 1D tensor of length B indicating, for each row in `seq_tensor`,
    the position of the first <eos> token if it exists,
    or the last token position if no <eos> is found.
    """
    B, T = seq_tensor.size()
    eos_mask = (seq_tensor == eos_id)  # (B, T) of booleans

    # Default: use the last token index (T-1) if no <eos> is found
    eos_pos = torch.full((B,), fill_value=(T - 1), dtype=torch.long, device=seq_tensor.device)

    # For each sequence, if we find any <eos>, set eos_pos to the first occurrence
    for i in range(B):
        indices = torch.where(eos_mask[i])[0]
        if len(indices) > 0:
            eos_pos[i] = indices[0].item()  # first <eos>

    return eos_pos



# ----------------------------
# Argument Parsing
# ----------------------------


# Our token dictionary (including our special separator token "1")
token_dict = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "A": 4, "B": 5, "C": 6, "D": 7, "E": 8, "F": 9,
    "G": 10, "H": 11, "I": 12, "J": 13, "K": 14, "L": 15,
    "M": 16, "N": 17, "O": 18, "P": 19, "Q": 20, "R": 21,
    "S": 22, "T": 23, "U": 24, "V": 25, "W": 26, "X": 27,
    "Y": 28, "Z": 29, "1": 30, "2": 31
}
# Inverse dictionary:
token_dict_inv = {v: k for k, v in token_dict.items()}

class ProteinTokenizer:
    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.inv_token_dict = {v: k for k, v in token_dict.items()}
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_id = token_dict[self.pad_token]
        self.bos_id = token_dict[self.bos_token]
        self.eos_id = token_dict[self.eos_token]
    def tokenize(self, sequence):
        return list(sequence)
    def convert_tokens_to_ids(self, tokens):
        return [self.token_dict.get(t, self.token_dict[self.unk_token]) for t in tokens]
    def encode(self, sequence, add_special_tokens=True):
        tokens = self.tokenize(sequence)
        if add_special_tokens:
            # For single‑protein examples we add both bos and eos.
            return self.convert_tokens_to_ids([self.bos_token] + tokens + [self.eos_token])
        else:
            return self.convert_tokens_to_ids(tokens)
    def decode(self, token_ids):
        return "".join(self.inv_token_dict.get(tid, self.unk_token) for tid in token_ids)
    def pad_sequences(self, sequences, padding_value=None, block_size=None):
        if block_size is None:
            block_size = max(len(seq) for seq in sequences)
        if padding_value is None:
            padding_value = self.token_dict[self.pad_token]
        return [seq[:block_size] + [padding_value] * max(0, block_size - len(seq)) for seq in sequences]

tokenizer = ProteinTokenizer(token_dict)



parser = argparse.ArgumentParser(description="IF")

# Training Arguments


parser.add_argument('--folder', type=str, default="/scratch/mnaseri1/img2", help='Path to training images folder')
parser.add_argument('--valid_frac', type=float, default=0, help='Validation fraction')
parser.add_argument('--encoder_model_dir', type=str, default="/home/mnaseri1/vqgan2/res_final", help='Folder to save results')
parser.add_argument('--amp', type=bool, default=True, help='Use Automatic Mixed Precision')
parser.add_argument('--gpt_model_dir', type=str, default="Generator_IF/out_pretrain_GPT_lora", help='Folder to save results')
parser.add_argument('--init_from', type=str, choices=['scratch', 'resume'], default='resume', help='Ini tialize from scratch or resume.')
parser.add_argument('--eval_only', default='False', help='Run evaluation only.')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value.')
parser.add_argument('--backend', type=str, default='nccl', help='Backend for distributed training.')
parser.add_argument('--out_dir', type=str, default='out_pretrain_GPT_lora2', help='Directory to save outputs and checkpoints.')
parser.add_argument('--dtype', type=str, choices=['float32', 'bfloat16', 'float16'], default='bfloat16', help='Data type for model weights.')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
parser.add_argument('--grad_accum_steps', type=int, default=2, help='Gradient accumulation steps.')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay factor.')
parser.add_argument('--lr', type=float, default=6e-4, help='Learning rate.')
parser.add_argument('--min_lr', type=float, default=5e-6, help='Minimum learning rate.')
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer.')
parser.add_argument('--beta2', type=float, default=0.95, help='Beta2 for Adam optimizer.')
parser.add_argument('--warmup_iters', type=int, default=2000, help='Number of warmup iterations.')
parser.add_argument('--lr_decay_iters', type=int, default=40000, help='Number of iterations for learning rate decay.')
parser.add_argument('--eval_interval', type=int, default=200, help='Iterations between evaluations.')
parser.add_argument('--eval_iters', type=int, default=100, help='Number of evaluation iterations.')
parser.add_argument('--always_save_checkpoint', default=True, help='Always save checkpoint.')
parser.add_argument('--log_interval', type=int, default=1, help='Iterations between logging.')
parser.add_argument('--max_iter', type=int, default=300000, help='Maximum number of training iterations.')
    
    # Distributed training arguments
parser.add_argument('--ddp', default = True, help='Use Distributed Data Parallel (DDP).')


args = parser.parse_args()



if args.ddp:
    init_process_group(backend=args.backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device)
    master = ddp_rank == 0
    seed_offset = ddp_rank
    print0(f"Initialized DDP: rank={ddp_rank}, local_rank={ddp_local_rank}, world_size={ddp_world_size}")
else:
    master = True
    seed_offset = 0
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print0(f"Single-process setup. Rank: {ddp_rank}/{ddp_world_size}, Device: {device}")


if master:
    os.makedirs(args.out_dir, exist_ok=True)
    print0(f"Results will be saved at {args.out_dir}!")


if master:
    wandb.init(project="IF_training",
               name="IF_training",
               config=vars(args)) 

# ----------------------------
# Set Seeds and CUDA Configurations
# ----------------------------

torch.manual_seed(2001 + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2001 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True      # Allow TF32 on cuDNN

device_type = "cuda" if "cuda" in device else "cpu"


ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == "cpu" else autocast(enabled=args.amp, dtype=ptdtype, device_type="cuda")

from dataclasses import dataclass
@dataclass
class ModelArgs:
    """
    Hyperparameters for the model.
    Adjust to your preference. Typically n_embd ~ 256..768,
    n_head ~ 4..16, and so on, depending on memory and resolution.
    """
    # Basic
    n_embd: int = 1024
    mlp_hidden_dim: int = int(1024 * 2)
    n_head: int = 16
    latent_dim: int = 256   # dimension for Q/K/V
    rope_dim: int = 8
    is_causal: bool = False  # Usually no causal mask for VQ-VAE

    # Codebook
    q_channels: int = 1024
    codebook_dim: int = 1024
    codebook_size: int = 4096

    # Depth
    n_layer_encoder: int = 2
    n_layer_decoder: int = 2
    

    beta: float = 0.25


scaling_rates =  [2, 2, 2, 2, 2, 2]

model_args_encoder = dict(
    n_layer_encoder=ModelArgs.n_layer_encoder,
    n_layer_decoder=ModelArgs.n_layer_decoder,
    n_head=ModelArgs.n_head,
    n_embd=ModelArgs.n_embd,
    is_causal = ModelArgs.is_causal,
    rope_dim = ModelArgs.rope_dim,
    mlp_hidden_dim = ModelArgs.mlp_hidden_dim,
    latent_dim = ModelArgs.latent_dim,
    q_channels=ModelArgs.q_channels,
    codebook_dim=ModelArgs.codebook_dim,
    codebook_size = ModelArgs.codebook_size,
    beta = ModelArgs.beta
)



@dataclass
class GPT_Config:
    n_embd: int = 1024
    latent_dim: int = 256
    max_seq_len: int = 1024
    n_head: int = 16
    n_layer: int = 32
    vocab_size: int = 32            
    ignore_index: int = -100
    block_size: int = max_seq_len
    seq_padding_idx: int = 0
    is_causal: bool = True
    rope_dim: int = 8
    mlp_hidden_dim: int = n_embd * 2


model_args_gpt = dict(
    n_layer=GPT_Config.n_layer,
    n_head=GPT_Config.n_head,
    n_embd=GPT_Config.n_embd,
    vocab_size=GPT_Config.vocab_size,
    seq_padding_idx = GPT_Config.seq_padding_idx,
    block_size=GPT_Config.block_size,
    max_seq_len=GPT_Config.max_seq_len,
    ignore_index = GPT_Config.ignore_index,
    is_causal = GPT_Config.is_causal,
    rope_dim = GPT_Config.rope_dim,
    mlp_hidden_dim = GPT_Config.mlp_hidden_dim,
    latent_dim = GPT_Config.latent_dim

)


best_val_loss = 1e9
print0(model_args_encoder)

print(f"Loading encoder model from {args.encoder_model_dir}")

ckpt_path = os.path.join("/home/xwang213/Mahan/IF/ckpt_vqvae37500.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
config = ModelArgs(**model_args_encoder)
encoder_model = Encoder(config, scaling_rates)
    
    # Load state dict with strict=False to allow missing keys (new modalities)
state_dict = checkpoint["model"]
encoder_state_dict = {k: v for k, v in state_dict.items() if k.startswith("encoders.")}
    # Initialize missing layers (new modalities) if necessary
missing_keys, unexpected_keys = encoder_model.load_state_dict(encoder_state_dict, strict=False)
if missing_keys:
        print(f"Missing keys after loading state dict: {missing_keys}")
        print("These are likely the new modality layers and will be randomly initialized.")
if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
        # These are the new layers; they are already initialized in the model's __init__
encoder_model.to(device)
    # Load optimizer state dict
encoder_model.eval()
print("Encoder Model Loaded")
iter_num = 0

print(f"Loading GPT Model from {args.gpt_model_dir}")
ckpt_path = os.path.join("/home/xwang213/Mahan/IF/ckpt_lora (2).pt")
checkpoint = torch.load(ckpt_path, map_location=device)
config = GPT_Config(**model_args_gpt)
gpt_model = GPT(config)
    
    # Load state dict with strict=False to allow missing keys (new modalities)
state_dict = checkpoint["model"]

    # Initialize missing layers (new modalities) if necessary
missing_keys, unexpected_keys = gpt_model.load_state_dict(state_dict, strict=False)
if missing_keys:
        print(f"Missing keys after loading state dict: {missing_keys}")
        print("These are likely the new modality layers and will be randomly initialized.")
if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
        # These are the new layers; they are already initialized in the model's __init__
gpt_model.to(device)




if args.init_from == "scratch":
    print("Trainig new model form scratch!")
    model = IFModel2(gpt_model=gpt_model, encodr_model=encoder_model, config=GPT_Config)
    model.to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type = device_type)
    iter_num = 0


if args.init_from == "resume":
    print(f"Resuming training from {args.out_dir}")
    ckpt_path = os.path.join(args.out_dir, "ckpt_IF2.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = IFModel2(gpt_model=gpt_model, encodr_model=encoder_model, config=GPT_Config)
    model.to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type = device_type)
    
    # Load state dict with strict=False to allow missing keys (new modalities)
    state_dict = checkpoint["model"]
    
    # Initialize missing layers (new modalities) if necessary
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys after loading state dict: {missing_keys}")
        print("These are likely the new modality layers and will be randomly initialized.")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
        # These are the new layers; they are already initialized in the model's __init__
    model.to(device)
    # Load optimizer state dict
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (args.beta1, args.beta2), device_type = device_type)
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    iter_num = checkpoint["iter_num"]
    #best_val_loss = checkpoint["best_val_loss"]

print0("World size:", ddp_world_size)
scaler = torch.GradScaler(enabled = args.amp)

checkpoint = None

if args.ddp:
    model = DDP(model, device_ids = [ddp_local_rank], find_unused_parameters=False)

print0("World size:", ddp_world_size)
scaler = torch.GradScaler(enabled = args.amp)


def get_lr(it, warmup_iters = 2000, lr = 6e-4, min_lr = 5e-6):
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    
    if it > args.lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (args.lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)



print0("Initializing Datasets and DataLoaders...")

root_folder = Path(args.folder)


t0 = time.time()
raw_model = model.module if args.ddp else model

    # Start from smallest bin => bin_40_64 => do 4000 steps => next bin => ...

dl, val_dl = train_for_all(
            root_folder,
            patch_size = scaling_rates[0],
            batch_size = args.batch_size,
            ddp = args.ddp,
            ddp_world_size = ddp_world_size,
            ddp_rank = ddp_rank,
            num_workers = 0,
            valid_frac = 0.0
        )
    
iter_num = 20000

def compute_sequence_identity(original, generated):
    """
    Compute sequence identity (%) between original and generated sequences.
    
    original: tensor of token IDs (ground truth sequence)
    generated: tensor of token IDs (generated sequence)
    
    Returns: sequence identity as a percentage.
    """
    original = original.squeeze().tolist()
    generated = generated.squeeze().tolist()
    
    min_length = min(len(original), len(generated))
    
    # Compute how many residues match in the overlapping region
    matches = sum(1 for a, b in zip(original[:min_length], generated[:min_length]) if a == b)
    
    # Normalize by the original sequence length
    seq_identity = (matches / len(original)) * 100
    return seq_identity


print0("Tokenizing training")

while True:
    micro_step = 0

    optimizer.zero_grad(set_to_none = True)

    for Y, X, X1, Z in dl:
        X = X.to(device)
        X1 = X1.to(device)
        Y = Y.to(device)


        
        with ctx:
            logits, loss = model(seq = X, img = Y, targets = X1)
            loss = loss / args.grad_accum_steps

        scaler.scale(loss).backward()

        micro_step += 1

        if micro_step % args.grad_accum_steps == 0:
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if args.ddp:
                model.require_backward_grad_sync = True

            if args.grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none = True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            if iter_num % args.log_interval == 0 and master:
                lossf = loss.item() * args.grad_accum_steps
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")
                wandb.log({
                    "train_loss": lossf,
                    "learning_rate": lr,
                    "time_ms": dt * 1000
                }, step=iter_num)

            iter_num += 1

            if (iter_num % args.eval_interval == 0 or iter_num == 1) and master :

                if args.always_save_checkpoint:
                    print(f"saving checkpoint to {args.out_dir}")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                    }
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt_IF2.pt'))
                    
            if (iter_num % args.eval_interval == 0 or iter_num == 1):
                print("Generating samples...")
                with torch.no_grad():
                    with ctx:
                        for i in range(2):
                            generated_cond = raw_model.generate_IF(
                                prefix = X[i:i+1, :1],
                                img = Y[i:i+1, :, :, :],
                                max_size = get_eos_positions(X[i:i+1, :], tokenizer.eos_id),
                                temperature = 0.8,
                                top_k = 5)
                            generated_cond_text = tokenizer.decode(generated_cond[0].cpu().tolist())
                            print(f"Conditional Sample for img {Z[i]}, generated_size = {len(generated_cond_text)}:\n{generated_cond_text}\n---------------")
                print("Generating GREEDY samples...")
                with torch.no_grad():
                    with ctx:
                        for i in range(2):
                            generated_cond = raw_model.generate_IF(
                                prefix = X[i:i+1, :1],
                                img = Y[i:i+1, :, :, :],
                                max_size = get_eos_positions(X[i:i+1, :], tokenizer.eos_id),
                                temperature = 0.9,
                                top_k = 1)
                            generated_cond_text = tokenizer.decode(generated_cond[0].cpu().tolist())
                            seq_identity = compute_sequence_identity(X[i], generated_cond[0])
                            print(f"Conditional greedy Sample for img {Z[i]}, generated_size = {len(generated_cond_text)}:\n{generated_cond_text}\n---------------")
                            print(f"Sequence Identity: {seq_identity:.2f}%")
                        
                
            
            if iter_num >= args.max_iter:
                break
        
if args.ddp:
    destroy_process_group()

