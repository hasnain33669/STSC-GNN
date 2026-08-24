import subprocess
import sys

def run_all_variants():
    variants = ['gcn', 'gat', 'gin']
    
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Running STSC-GNN ({variant.upper()})")
        print('='*60)
        
        cmd = [sys.executable, f'scripts/run_{variant}.py']
        subprocess.run(cmd, check=True)
    
    print("\n✅ All variants completed successfully!")

if __name__ == "__main__":
    run_all_variants()
