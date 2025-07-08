import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import h5py
import os
import itertools
from tqdm import tqdm
import argparse
import gc

class YearPairDataset(Dataset):
    def __init__(self, year_pairs, tokenizer, prompt_template):
        self.year_pairs = year_pairs
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        
    def __len__(self):
        return len(self.year_pairs)
    
    def __getitem__(self, idx):
        year1, year2 = self.year_pairs[idx]
        prompt = self.prompt_template.format(year1=year1, year2=year2)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=256,
            truncation=True
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'year_pair': (year1, year2),
            'pair_idx': idx
        }

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def extract_residuals_worker(rank, world_size, args):
    setup(rank, world_size)
    
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{rank}",
        trust_remote_code=True
    )
    
    model = torch.compile(model)
    model.eval()
    
    years = list(range(1525, 2525))
    all_pairs = list(itertools.combinations(years, 2))
    self_pairs = [(y, y) for y in years]
    all_pairs.extend(self_pairs)
    
    if rank == 0:
        print(f"Total unique pairs (no symmetries): {len(all_pairs)}")
    
    prompt_template = """How close are the two years on a scale of 0 (completely dissimilar) to 1 (completely similar)?

Respond only with the rating.

Year_1: {year1}

Year_2: {year2}

Rating:"""
    
    dataset = YearPairDataset(all_pairs, tokenizer, prompt_template)
    
    all_layer_indices = list(range(model.config.num_hidden_layers))
    if args.layers_to_save:
        layers_to_save = [int(l) for l in args.layers_to_save]
        if not all(l in all_layer_indices for l in layers_to_save):
            raise ValueError(f"Invalid layer index provided. Available layers are from 0 to {len(all_layer_indices)-1}.")
    else:
        layers_to_save = all_layer_indices

    if rank == 0:
        print(f"Saving residuals for layers: {layers_to_save}")
    
    hidden_size = model.config.hidden_size
    
    samples_per_rank = len(all_pairs) // world_size
    rank_start_idx = rank * samples_per_rank
    rank_end_idx = (rank + 1) * samples_per_rank if rank != world_size - 1 else len(all_pairs)
    num_samples_for_rank = rank_end_idx - rank_start_idx

    chunk_size = args.chunk_size
    num_chunks = (num_samples_for_rank + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        
        expected_chunk_filename = f"{args.output_dir}/residuals_rank_{rank}_chunk_{chunk_idx}.h5"
        
        if os.path.exists(expected_chunk_filename):
            try:
                with h5py.File(expected_chunk_filename, 'r') as f:
                    
                    if 'pair_indices' in f:
                        if rank == 0:
                            print(f"Rank {rank}, Chunk {chunk_idx}: Output file found and seems valid. Skipping.")
                        continue
            except Exception as e:
                
                if rank == 0:
                    print(f"Rank {rank}, Chunk {chunk_idx}: Found corrupt file '{expected_chunk_filename}'. Re-processing. Error: {e}")

        chunk_start_in_rank = chunk_idx * chunk_size
        chunk_end_in_rank = min(chunk_start_in_rank + chunk_size, num_samples_for_rank)
        
        chunk_abs_start = rank_start_idx + chunk_start_in_rank
        chunk_abs_end = rank_start_idx + chunk_end_in_rank
        
        chunk_indices = range(chunk_abs_start, chunk_abs_end)
        chunk_dataset = Subset(dataset, chunk_indices)
        
        chunk_dataloader = DataLoader(
            chunk_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        chunk_residuals = {f'layer_{i}': [] for i in layers_to_save}
        
        with torch.no_grad():
            for batch in tqdm(chunk_dataloader, desc=f"Rank {rank}, Chunk {chunk_idx+1}/{num_chunks}", disable=(rank!=0)):
                input_ids = batch['input_ids'].to(rank, non_blocking=True)
                attention_mask = batch['attention_mask'].to(rank, non_blocking=True)
                
                residuals = {}
                
                def get_activation(name):
                    def hook(model, input, output):
                        hidden_states = output[0]
                        last_positions = attention_mask.sum(dim=1) - 1
                        batch_residuals_list = []
                        for i, pos in enumerate(last_positions):
                            batch_residuals_list.append(hidden_states[i, pos, :].cpu().half().numpy())
                        residuals[name] = np.stack(batch_residuals_list)
                    return hook
                
                handles = []
                for i, layer in enumerate(model.model.layers):
                    if i in layers_to_save:
                        handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
                        handles.append(handle)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                for layer_name, activations in residuals.items():
                    chunk_residuals[layer_name].append(activations)
                
                for handle in handles:
                    handle.remove()
                
                torch.cuda.empty_cache()
        
        final_chunk_residuals = {}
        for layer_name in chunk_residuals:
            if chunk_residuals[layer_name]:
                final_chunk_residuals[layer_name] = np.concatenate(chunk_residuals[layer_name], axis=0)
            else:
                final_chunk_residuals[layer_name] = np.empty((0, hidden_size), dtype=np.float16)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        with h5py.File(expected_chunk_filename, 'w') as f:
            for layer_name, activations in final_chunk_residuals.items():
                f.create_dataset(layer_name, data=activations, compression='gzip')
            
            f.create_dataset('pair_indices', data=np.array(list(chunk_indices)))

        if rank == 0:
            print(f"Rank {rank}: Saved chunk {chunk_idx+1}/{num_chunks} to {expected_chunk_filename}")
        
        del chunk_residuals, final_chunk_residuals
        gc.collect()
        torch.cuda.empty_cache()
    
    dist.barrier()
    
    if rank == 0:
        print("\n" + "="*50)
        print("Residual extraction complete!")
        print(f"Output files are saved in: {args.output_dir}")
        print("Each rank has saved its own chunk files. No merging is performed.")
        print("="*50 + "\n")
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description="Extract residual stream activations from a model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output HDF5 files.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=10000, help="Number of SAMPLES per chunk file, not batches.")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use.")
    parser.add_argument("--layers_to_save", nargs='+', default=None, help="Optional list of layer indices to save (e.g., 0 15 31). Saves all if not specified.")    
    
    args = parser.parse_args()    
    os.makedirs(args.output_dir, exist_ok=True)    
    mp.spawn(extract_residuals_worker, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()
