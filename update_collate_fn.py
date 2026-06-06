import json

improved_custom_collate_fn = '''def custom_collate_fn(batch):
    """
    Custom collate function that robustly handles None values, validates tensors,
    and filters invalid samples from the batch.
    """
    valid_batch = []
    
    for sample_idx, sample in enumerate(batch):
        try:
            # Skip None samples
            if sample is None:
                print(f"[Collate] Skipping None sample at index {sample_idx}")
                continue
            
            # Verify sample is a dict
            if not isinstance(sample, dict):
                print(f"[Collate] Skipping non-dict sample at index {sample_idx}")
                continue
            
            # Check each field for None values
            invalid_fields = [k for k, v in sample.items() if v is None]
            if invalid_fields:
                print(f"[Collate] Skipping sample {sample_idx} - None fields: {invalid_fields}")
                continue
            
            # Validate all values are tensors and non-empty
            for key, value in sample.items():
                if not isinstance(value, torch.Tensor):
                    print(f"[Collate] Warning: {key} is {type(value)}, converting...")
                    if isinstance(value, np.ndarray):
                        value = torch.from_numpy(value)
                    else:
                        raise ValueError(f"Cannot convert {key} to tensor")
                    sample[key] = value
                
                # Check tensor is not empty or NaN
                if value.numel() == 0:
                    raise ValueError(f"Empty tensor for {key}")
                if torch.isnan(value).any():
                    print(f"[Collate] Warning: {key} contains NaN, replacing with zeros")
                    sample[key] = torch.zeros_like(value)
            
            valid_batch.append(sample)
        
        except Exception as e:
            print(f"[Collate] Error processing sample {sample_idx}: {e}")
            continue
    
    # Handle empty batch
    if len(valid_batch) == 0:
        print("[Collate] Error: All samples in batch were invalid!")
        return None
    
    # Collate valid samples
    result = {}
    keys = valid_batch[0].keys()
    
    for key in keys:
        try:
            values = [sample[key] for sample in valid_batch]
            
            # Check all values are tensors
            if not all(isinstance(v, torch.Tensor) for v in values):
                raise ValueError(f"Not all values for {key} are tensors")
            
            # Check shapes are compatible
            shapes = [v.shape for v in values]
            if len(set(str(s[1:]) for s in shapes)) > 1:
                # Different shapes (except batch dim) - pad or handle
                print(f"[Collate] Warning: {key} has mixed shapes: {shapes}")
            
            # Stack tensors
            result[key] = torch.stack(values, dim=0)
        
        except Exception as e:
            print(f"[Collate] Error stacking {key}: {e}")
            # Fallback to list
            result[key] = values
    
    return result'''

def update_notebooks():
    notebooks = [
        'SkinLesionClassifier_V10_Final_1_cell_by_cell.ipynb',
        'SkinLesionClassifier_V10_Final_cell_by_cell.ipynb'
    ]
    
    for nb_file in notebooks:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        modified = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell.get('source', []))
                if 'def custom_collate_fn' in source:
                    # Replace entire function
                    cell['source'] = improved_custom_collate_fn.split('\n')
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
                    modified = True
                    print(f"✓ Updated custom_collate_fn in {nb_file}")
                    break
        
        if modified:
            with open(nb_file, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)

if __name__ == '__main__':
    update_notebooks()
