# STSC-GNN: Sequence-Derived Topology-Aware Graph Neural Network

STSC-GNN is a unified framework for solvent-conditioned prediction of peptide structural composition. It integrates:
- Sequence-derived residue graphs with multi-scale topology
- ESM-2 protein language model embeddings
- Physicochemical amino acid features
- Continuous solvent descriptors
- Bidirectional peptide-solvent cross-attention

## Variants

This repository provides three GNN backbone implementations:

| **STSC-GNN (GCN)** 
| **STSC-GNN (GAT)** 
| **STSC-GNN (GIN)** 

## Installation
setup(

    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "rdkit>=2023.03.0",
        "fair-esm>=2.0.0",
        "transformers>=4.30.0",
        "torch_geometric>=2.3.0",
        "umap-learn>=0.5.0",
        "biopython>=1.81",
        "pyyaml>=6.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
    ],
    python_requires=">=3.9",
)

# Install dependencies
pip install -r requirements.txt
