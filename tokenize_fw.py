import torch 
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm, trange
from huggingface_hub import hf_api
from tempfile import NamedTemporaryFile


model_id = "Qwen/Qwen2.5-Math-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id, 
#     device_map="cuda",
#     torch_dtype="auto",
#     attn_implementation="flash_attention_2",
#     )

def tokenize_and_truncate(formatted_instances):
    tokenized = tokenizer(formatted_instances, truncation=True, padding=True, max_length=1024, return_tensors="pt")
    detokenized = tokenizer.batch_decode(tokenized["input_ids"])
    return tokenized, detokenized

def sample_half_toks(fw_raw_dataset, max_num_tokens):
    raw_instances = []
    formatted_conversations = []
    tokenized_conversations = []
    num_tokens = 0
    ds_columns = fw_raw_dataset.column_names
    # formatted_ds = or_raw_dataset.map(lambda x: {"formatted_chat" : tokenizer.apply_chat_template(x["messages"], tokenize=False, add_generation_prompt=False)}, num_proc=16, remove_columns=ds_columns)
    progress_bar = tqdm(total=max_num_tokens, desc="Tokens processed", unit="token")
    for i in range(0, len(fw_raw_dataset), 100):
        instances = fw_raw_dataset[i:min(i+100, len(fw_raw_dataset))]["text"]
        # formatted_instances = format_or_prompts(raw_instances)
        tokenized, detokenized = tokenize_and_truncate(instances)
        len_tokenized = tokenized["input_ids"].shape[0] * tokenized["input_ids"].shape[1]
        if num_tokens + len_tokenized > max_num_tokens:
            break
        raw_instances.extend(instances)
        formatted_conversations.extend(detokenized)
        tokenized_conversations.extend(tokenized["input_ids"])
        num_tokens += len_tokenized
        progress_bar.update(len_tokenized)
    progress_bar.close()
    return raw_instances, formatted_conversations, tokenized_conversations, num_tokens

def create_dataset(raw_instances, formatted_conversations, tokenized_conversations):
    assert len(raw_instances) == len(formatted_conversations) == len(tokenized_conversations)
    # turn tokenized from list of tensors to tensor here:
    tokenized_conversations_tensor = torch.stack(tokenized_conversations)

    dataset_list = [
        {
            "raw": raw,
            "formatted": formatted,
            "tokenized": tokenized,
        }
        for raw, formatted, tokenized in zip(
            raw_instances, formatted_conversations, tokenized_conversations_tensor
        )
    ]
    return Dataset.from_list(dataset_list)

@torch.no_grad
def main():
    tokens_from_openr1 = 95_982_592
    print(f"Setting max_num_tokens to {tokens_from_openr1}")

    fw_raw_dataset = load_dataset(
        # "HuggingFaceFW/fineweb", 
        "science-of-finetuning/fineweb-1m-sample",
        # name="sample-10BT", 
        split="train", 
        # streaming=True
        )
    raw_instances, formatted_conversations, tokenized_conversations, num_tokens = sample_half_toks(fw_raw_dataset, tokens_from_openr1)

    dataset = create_dataset(raw_instances, formatted_conversations, tokenized_conversations)
 

    print(f"Generated data with {num_tokens} tokens")
    print(f"Saving generated data to disk")

    dataset.save_to_disk("/home/user/repos/R1-crosscoder/data/fineweb-qwen-2.5-math-1.5b")
    dataset.push_to_hub("Neelectric/fineweb-qwen-2.5-math-1.5b")


if __name__ == "__main__":
    main()

