import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import h5py
import os
import argparse
from tqdm import tqdm
import json
from typing import Dict, List, Tuple
import gc
from contextlib import contextmanager

class ActivationCollector:
    
    def __init__(self, model, layer_indices=None):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.layer_indices = layer_indices or list(range(len(model.model.layers)))
        self._register_hooks()
    
    def _register_hooks(self):
        
        def get_activation_hook(layer_idx):
            def hook(module, input, output):
                
                if isinstance(output, tuple):
                    activation = output[0][:, -1, :].detach().cpu()
                else:
                    activation = output[:, -1, :].detach().cpu()
                self.activations[f'layer_{layer_idx}_ffn'] = activation
            return hook
        
        for layer_idx in self.layer_indices:
            
            layer = self.model.model.layers[layer_idx]
            if hasattr(layer, 'mlp'):
                
                hook = layer.mlp.register_forward_hook(get_activation_hook(layer_idx))
                self.hooks.append(hook)
    
    def clear_activations(self):
        self.activations.clear()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    if local_rank not in [-1, 0]:
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:
        dist.barrier(device_ids=[0])

def load_model_and_tokenizer(model_path, rank):
    print(f"Rank {rank}: Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=f"cuda:{rank}",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    model.eval()
    return model, tokenizer

def generate_year_inputs(start_year=1525, end_year=2524):
    years = list(range(start_year, end_year + 1))
    
    temporal_inputs = [f"Year: {year}" for year in years]
    
    numerical_inputs = [f"Number: {year}" for year in years]
    
    return years, temporal_inputs, numerical_inputs

def collect_activations_batch(model, tokenizer, collector, inputs, batch_size=32):
    all_activations = {}
    
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size), desc="Processing batches"):
            batch_inputs = inputs[i:i + batch_size]
            
            encoded = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=32
            )
            
            input_ids = encoded['input_ids'].to(model.device)
            attention_mask = encoded['attention_mask'].to(model.device)
            collector.clear_activations()
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            for layer_name, activation in collector.activations.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = []
                all_activations[layer_name].append(activation.clone())
            
            del input_ids, attention_mask
            torch.cuda.empty_cache()
    
    for layer_name in all_activations:
        all_activations[layer_name] = torch.cat(all_activations[layer_name], dim=0)
    
    return all_activations

def save_activations_distributed(activations, years, condition, rank, world_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/activations_{condition}_rank_{rank}.h5"
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('years', data=years)
        
        for layer_name, activation in activations.items():
            f.create_dataset(layer_name, data=activation.numpy())
        f.attrs['condition'] = condition
        f.attrs['rank'] = rank
        f.attrs['world_size'] = world_size
        f.attrs['num_years'] = len(years)
        f.attrs['activation_shape'] = str(list(activation.shape))

def merge_activations_from_ranks(output_dir, condition, world_size):

    print(f"Merging activations for condition: {condition}")    
    merged_filename = f"{output_dir}/activations_{condition}_merged.h5"
    first_file = f"{output_dir}/activations_{condition}_rank_0.h5"
    with h5py.File(first_file, 'r') as f:
        years = f['years'][:]
        layer_names = [key for key in f.keys() if key.startswith('layer_')]
        activation_shape = f[layer_names[0]].shape
    
    with h5py.File(merged_filename, 'w') as merged_f:
        merged_f.create_dataset('years', data=years)
        
        for layer_name in layer_names:
            merged_shape = (activation_shape[0], activation_shape[1])
            merged_f.create_dataset(layer_name, shape=merged_shape, dtype=np.float16)
        
        for rank in range(world_size):
            rank_file = f"{output_dir}/activations_{condition}_rank_{rank}.h5"
            with h5py.File(rank_file, 'r') as rank_f:
                for layer_name in layer_names:
                    merged_f[layer_name][:] = rank_f[layer_name][:]
        merged_f.attrs['condition'] = condition
        merged_f.attrs['num_years'] = len(years)
        merged_f.attrs['world_size'] = world_size

def worker_process(rank, world_size, model_path, output_dir, start_year, end_year, batch_size):
    
    try:
        
        setup_distributed(rank, world_size)        
        model, tokenizer = load_model_and_tokenizer(model_path, rank)        
        collector = ActivationCollector(model)        
        years, temporal_inputs, numerical_inputs = generate_year_inputs(start_year, end_year)        
        print(f"Rank {rank}: Processing {len(years)} years...")
        print(f"Rank {rank}: Collecting temporal activations...")
        temporal_activations = collect_activations_batch(
            model, tokenizer, collector, temporal_inputs, batch_size
        )
        save_activations_distributed(
            temporal_activations, years, "temporal", rank, world_size, output_dir
        )
        del temporal_activations
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Rank {rank}: Collecting numerical activations...")
        numerical_activations = collect_activations_batch(
            model, tokenizer, collector, numerical_inputs, batch_size
        )
        save_activations_distributed(
            numerical_activations, years, "numerical", rank, world_size, output_dir
        )
        collector.remove_hooks()
        del numerical_activations, model, tokenizer
        torch.cuda.empty_cache()
        dist.barrier()
        if rank == 0:
            merge_activations_from_ranks(output_dir, "temporal", world_size)
            merge_activations_from_ranks(output_dir, "numerical", world_size)
            print("All activations collected and merged")        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise e
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Collect neural activations from LLM")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("--output_dir", type=str, default="./neural_activations",
                       help="Output directory for activation data")
    parser.add_argument("--start_year", type=int, default=1525,
                       help="Start year for analysis")
    parser.add_argument("--end_year", type=int, default=2524,
                       help="End year for analysis")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--world_size", type=int, default=8,
                       help="Number of GPUs to use")
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = {
        "model_path": args.model_path,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "batch_size": args.batch_size,
        "world_size": args.world_size
    }
    
    with open(f"{args.output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Starting neural activation collection with {args.world_size} GPUs...")
    print(f"Model: {args.model_path}")
    print(f"Year range: {args.start_year} - {args.end_year}")
    print(f"Output directory: {args.output_dir}")
    
    mp.spawn(
        worker_process,
        args=(args.world_size, args.model_path, args.output_dir, 
              args.start_year, args.end_year, args.batch_size),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()
