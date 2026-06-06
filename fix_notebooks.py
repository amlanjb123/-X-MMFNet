import json
import os

def fix_notebook(notebook_path):
    """Add collate_fn=custom_collate_fn to DataLoader calls"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get('cells', []):
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            source_str = ''.join(source) if isinstance(source, list) else source
            
            # Check if this cell has DataLoader creations
            if 'DataLoader(' in source_str and 'custom_collate_fn' not in source_str:
                if 'num_workers=config.NUM_WORKERS, pin_memory=True)' in source_str:
                    # Replace all three occurrences
                    source_str = source_str.replace(
                        'num_workers=config.NUM_WORKERS, pin_memory=True)',
                        'num_workers=config.NUM_WORKERS, pin_memory=True, collate_fn=custom_collate_fn)'
                    )
                    modified = True
                    cell['source'] = source_str.split('\n')
                    # Add newline back
                    cell['source'] = [line + '\n' for line in cell['source'][:-1]] + [cell['source'][-1]]
    
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        return True
    return False

notebooks = [
    'SkinLesionClassifier_V10_Final_1_cell_by_cell.ipynb',
    'SkinLesionClassifier_V10_Final_cell_by_cell.ipynb'
]

for nb in notebooks:
    nb_path = os.path.join(os.path.dirname(__file__), nb)
    if os.path.exists(nb_path):
        if fix_notebook(nb_path):
            print(f"✓ Fixed {nb}")
        else:
            print(f"~ {nb} already has collate_fn or no DataLoaders found")
    else:
        print(f"✗ {nb} not found")
