import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_molecules_from_sdf(sdf_dir):
    """
    Load molecules from SDF files
    """
    sdf_files = list(Path(sdf_dir).glob("*.sdf"))
    molecules = []
    names = []
    
    for sdf_file in sdf_files:
        # Use RDKit's SDMolSupplier to read SDF files
        supplier = Chem.SDMolSupplier(str(sdf_file))
        for mol in supplier:
            if mol is not None:
                # Extract drug name from filename
                drug_name = str(sdf_file.stem).rsplit('_', 1)[0]  # Remove CID part
                molecules.append(mol)
                names.append(drug_name)
    
    return molecules, names

def calculate_fingerprints(molecules, fp_type='morgan', radius=2, n_bits=2048):
    """
    Calculate molecular fingerprints for clustering
    """
    fingerprints = []
    
    for mol in molecules:
        if fp_type == 'morgan':
            # Use ECFP (Morgan) fingerprints as bit vectors
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=n_bits)
        elif fp_type == 'topological':
            fp = Chem.RDKitFingerprint(mol, maxPath=5, fpSize=n_bits)
        elif fp_type == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        fingerprints.append(fp)
    
    return fingerprints

def cluster_molecules_butina(fingerprints, cutoff=0.6):
    """
    Cluster molecules using Butina clustering algorithm
    """
    # Calculate Tanimoto similarity matrix
    from rdkit import DataStructs
    similarities = []
    for i in range(len(fingerprints)):
        for j in range(i+1, len(fingerprints)):
            sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append((i, j, sim))
    
    # Perform Butina clustering
    cluster_indices = Butina.ClusterData(
        similarities, 
        len(fingerprints), 
        cutoff, 
        isDistData=False,
        reordering=True
    )
    
    # Create cluster assignments
    cluster_assignments = [0] * len(fingerprints)
    for cluster_id, cluster in enumerate(cluster_indices):
        for idx in cluster:
            cluster_assignments[idx] = cluster_id
    
    return cluster_assignments, cluster_indices

def cluster_molecules_sklearn(fingerprints, method='agglomerative', n_clusters=50):
    """
    Cluster molecules using sklearn clustering methods
    """
    # Convert fingerprints to numpy array
    # First convert RDKit fingerprints to bit vectors for sklearn
    from rdkit import DataStructs
    fp_list = []
    
    for fp in fingerprints:
        # Convert to numpy array using RDKit's ConvertToNumpyArray
        arr = np.zeros((len(fp),))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
    
    fp_array = np.array(fp_list)
    
    if method == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    cluster_labels = clustering.fit_predict(fp_array)
    
    return cluster_labels

def analyze_chemical_groups(molecules):
    """
    Analyze common chemical groups in molecules
    """
    # Count functional groups using RDKit
    functional_groups = Counter()
    
    for mol in molecules:
        if mol is not None:
            # Calculate various molecular descriptors to identify functional groups
            mol_formula = rdMolDescriptors.CalcMolFormula(mol)
            functional_groups[mol_formula] += 1
            
            # Count rings
            sssr = mol.GetRingInfo().NumRings()
            if sssr > 0:
                functional_groups[f'Rings_{sssr}'] += 1
            
            # Count aromatic rings
            aromatic_rings = 0
            for ring in mol.GetRingInfo().AtomRings():
                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    aromatic_rings += 1
            if aromatic_rings > 0:
                functional_groups[f'Aromatic_Rings_{aromatic_rings}'] += 1
            
            # Identify common functional groups
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 8:  # Oxygen
                    # Check if it's a carbonyl, alcohol, or ether
                    if atom.GetFormalCharge() == 0:
                        functional_groups['Oxygen'] += 1
                elif atom.GetAtomicNum() == 7:  # Nitrogen
                    functional_groups['Nitrogen'] += 1
                elif atom.GetAtomicNum() == 16:  # Sulfur
                    functional_groups['Sulfur'] += 1
                elif atom.GetAtomicNum() == 15:  # Phosphorus
                    functional_groups['Phosphorus'] += 1
    
    return functional_groups

def visualize_clusters(molecules, names, cluster_labels, method='tsne'):
    """
    Visualize clusters using t-SNE or PCA
    """
    # Calculate fingerprints
    fingerprints = calculate_fingerprints(molecules)
    
    # Convert to numpy array
    from rdkit import DataStructs
    fp_list = []
    for fp in fingerprints:
        # Convert to numpy array using RDKit's ConvertToNumpyArray
        arr = np.zeros((len(fp),))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
    
    fp_array = np.array(fp_list)
    
    # Apply dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    coords = reducer.fit_transform(fp_array)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap='tab20', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(f'Molecular Clustering using {method.upper()}')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(f'clusters_{method}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close to avoid display issues in non-interactive environments

def main():
    print("Loading molecules from SDF files...")
    molecules, names = load_molecules_from_sdf('sdfs')
    print(f"Loaded {len(molecules)} molecules")
    
    # Filter out any None molecules
    valid_data = [(mol, name) for mol, name in zip(molecules, names) if mol is not None]
    molecules = [x[0] for x in valid_data]
    names = [x[1] for x in valid_data]
    print(f"Valid molecules after filtering: {len(molecules)}")
    
    # Calculate fingerprints
    print("Calculating fingerprints...")
    fingerprints = calculate_fingerprints(molecules, fp_type='morgan')
    
    # Perform clustering using Butina
    print("Performing Butina clustering...")
    cluster_labels, cluster_indices = cluster_molecules_butina(fingerprints, cutoff=0.6)
    
    print(f"Found {len(cluster_indices)} clusters")
    print("Largest clusters:")
    for i, cluster in enumerate(cluster_indices[:5]):  # Show 5 largest clusters
        cluster_mol_names = [names[idx] for idx in cluster]
        print(f"Cluster {i}: {len(cluster)} molecules - {cluster_mol_names[:5]}...")  # Show first 5 names
    
    # Alternative clustering with sklearn
    print("\nPerforming Agglomerative clustering...")
    sklearn_labels = cluster_molecules_sklearn(fingerprints, method='agglomerative', n_clusters=50)
    
    # Analyze chemical groups
    print("\nAnalyzing chemical groups...")
    chemical_groups = analyze_chemical_groups(molecules)
    
    print("\nMost common chemical groups:")
    for group, count in chemical_groups.most_common(10):
        print(f"{group}: {count}")
    
    # Visualize clusters
    print("Visualizing clusters...")
    visualize_clusters(molecules[:100], names[:100], cluster_labels[:100])  # Just first 100 for visualization
    
    # Calculate molecular descriptors to get more insight
    print("\nCalculating molecular descriptors for top clusters...")
    for i, cluster in enumerate(cluster_indices[:5]):
        print(f"\nCluster {i} (size: {len(cluster)}):")
        for idx in cluster[:3]:  # Show first 3 molecules in each cluster
            mol = molecules[idx]
            name = names[idx]
            weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            h_donor_count = Descriptors.NumHDonors(mol)
            h_acceptor_count = Descriptors.NumHAcceptors(mol)
            print(f"  - {name[:30]:25} MW: {weight:.1f}, LogP: {logp:.2f}, RotBonds: {rotatable_bonds}, HDonors: {h_donor_count}, HAcceptors: {h_acceptor_count}")

if __name__ == "__main__":
    main()