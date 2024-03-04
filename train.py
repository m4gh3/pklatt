import math
import gzip
import random
import tqdm
import numpy as np
from functools import wraps, partial

import torch
from torch.optim import Adam, AdamW
from torch import Tensor
from torch.nn import Module, functional as F
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator

from gateloop_transformer import Transformer

# constants

NUM_BATCHES = int(2.5e4) #int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-3 #8e-4
WEIGHT_DECAY = 1.25e-6 #0.
VALIDATE_EVERY = 100
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 256

WANDB = True
PROJECT_NAME = 'pklatt'
RUN_NAME = '2j'#'baseline gateloop'

# hf accelerate

accelerator = Accelerator(log_with = 'wandb' if WANDB else None)

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net: Module,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)
        logits = logits[:, -1]

        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample[..., None]), dim = -1)

    return out[..., prompt_seq_len:]

# optimizer

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr = 1e-4,
    wd = 0.,
    betas = (0.9, 0.99),
    eps = 1e-8,
    group_wd_params = True,
    **kwargs
):
    opt_kwargs = dict(lr = lr, betas = betas, eps = eps)

    if wd == 0:
        return Adam(params, **opt_kwargs)

    opt_kwargs = {'weight_decay': wd, **opt_kwargs}

    if not group_wd_params:
        return AdamW(params, **opt_kwargs)

    wd_params, no_wd_params = separate_weight_decayable_params(params)

    params = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(params, **opt_kwargs)

# instantiate transformer

hparams = dict(
    num_tokens = 256,
    dim = 256, #480, #320, #160,
    depth = 8,
    use_gate_looped_attn = True,
    gate_loop_heads = 16,              # in paper, they used heads == dim, but should experiment with less heads, as memory allows. we should figure out how much max-heads contributed to the performance, if any
    data_dependent_rel_pos = False,
    attn_softmax_normalize = True,
    ablate_complex = False,
    ablate_state_transition = False,
    rotary_emb = False,
    post_ln_norm = True
)

model = Transformer(**hparams)

# initialize experiment tracking

num_parameters = sum(p.numel() for p in model.parameters())
print(f'number of parameters: {num_parameters}')

wandb_config = {**hparams, 'num_parameters': num_parameters}
accelerator.init_trackers(PROJECT_NAME, config = wandb_config)

if WANDB and exists(RUN_NAME) and len(accelerator.trackers) > 0:
    accelerator.trackers[0].run.name = RUN_NAME

# prepare enwik8 data

with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# optimizer

optim = get_optimizer(
    model.parameters(),
    lr = LEARNING_RATE,
    wd = WEIGHT_DECAY
)

#schedulwee
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=200, factor=0.75 )

# prepare

(
    model,
    optim,
    train_loader,
    val_loader
) = accelerator.prepare(
    model,
    optim,
    train_loader,
    val_loader
)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training

for i in tqdm.tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = model(data, return_loss = True)

        accelerator.backward(loss / GRAD_ACCUM_EVERY)

    print(f"training loss: {loss.item():.3f}")
    accelerator.log(dict(loss = loss.item()), step = i)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    #scheduler.step(loss)
    optim.zero_grad()
    
    accelerator.wait_for_everyone()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = model(valid_data, return_loss = True)
            print(f"validation loss: {loss.item():.3f}")
            accelerator.log(dict(valid_loss = loss.item()), step = i)

    accelerator.wait_for_everyone()

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.to(accelerator.device)

        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        prompt = inp[None, ...]

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print("\n\n", base_decode_output, "\n")
