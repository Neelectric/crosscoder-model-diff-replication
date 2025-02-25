# %%
from utils import *
from trainer import Trainer
# %%
# device = 'cuda:0'
device = "cpu"
n_devices = 1
if torch.cuda.is_available():
    device = "cuda"
    n_devices = torch.cuda.device_count()
elif torch.backends.mps.is_available():
    device = "mps"

### Gemma ids
# base_model_id = "google/gemma-2-2b"
# ft_model_id = "google/gemma-2-2b-it"

### Qwen ids
base_model_id = "Qwen/Qwen2.5-Math-1.5B"
# ft_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ft_model_id = "Dongwei/Qwen2.5-1.5B-Open-R1-GRPO_Math"

all_tokens = load_pile_lmsys_mixed_tokens(base_model_id)

base_model = HookedTransformer.from_pretrained(
    base_model_id, 
    device=device, 
    n_devices=n_devices,
)

ft_model = HookedTransformer.from_pretrained(
    ft_model_id, 
    device=device, 
    n_devices=n_devices,
)

# %%

# %%
default_cfg = {
    "seed": 49,
    "batch_size": 8192, #originally 4096
    "buffer_mult": 128,
    "lr": 5e-5,
    "num_tokens": 400_000_000, #originally 400_000_000
    "l1_coeff": 2,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": base_model.cfg.d_model,
    "dict_size": 2**14,
    "seq_len": 1024,
    "enc_dtype": "fp32",
    "model_name": "qwen2.5-math-1.5b",
    "site": "resid_pre",
    "device": device,
    "model_batch_size": 4,
    "log_every": 3000,
    "save_every": 10, # originally 30000 
    "dec_init_norm": 0.08,
    "hook_point": "blocks.14.hook_resid_pre",
    "wandb_project": "R1-crosscoder",
    "wandb_entity": "Neelectric",
    "run_name": "qwen2.5-math-1.5b_crosscoder",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, ft_model, all_tokens)
trainer.train()
# %%