import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering
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
        supplier = Chem.SDMolSupplier(str(sdf_file))
        for mol in supplier:
            if mol is not None:
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
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol, maxPath=5, fpSize=n_bits)
        elif fp_type == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        fingerprints.append(fp)
    
    return fingerprints

def find_common_pharmacophores(molecules, names):
    """
    Identify common pharmacophore features in the dataset
    """
    # Use RDKit's pattern recognition to identify common structural features
    ring_systems = Counter()
    atom_types = Counter()
    functional_groups = Counter()
    
    for mol in molecules:
        if mol is not None:
            # Count ring systems
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            ring_systems[f"Rings_{num_rings}"] += 1
            
            # Count aromatic rings
            aromatic_rings = sum(1 for ring in ring_info.AtomRings() 
                                if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
            if aromatic_rings > 0:
                functional_groups[f"Aromatic_Rings_{aromatic_rings}"] += 1
            
            # Count atom types and functional groups
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                if atomic_num == 6:  # Carbon
                    atom_types['Carbon'] += 1
                elif atomic_num == 7:  # Nitrogen
                    atom_types['Nitrogen'] += 1
                    # Check for amine, amide, etc.
                    if atom.GetDegree() == 3 and atom.GetFormalCharge() == 0:
                        functional_groups['Amine'] += 1
                    elif atom.GetDegree() == 3 and atom.GetFormalCharge() == 1:
                        functional_groups['Ammonium'] += 1
                elif atomic_num == 8:  # Oxygen
                    atom_types['Oxygen'] += 1
                    # Check for hydroxyl, carbonyl, etc.
                    if atom.GetDegree() == 1:  # Likely carbonyl or hydroxyl
                        neighbors = list(atom.GetNeighbors())
                        if any(n.GetAtomicNum() == 6 for n in neighbors):  # bonded to carbon
                            if any(mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE 
                                   for n in neighbors):
                                functional_groups['Carbonyl'] += 1
                            else:
                                functional_groups['Hydroxyl'] += 1
                    elif atom.GetDegree() == 2:  # Ester or ether
                        functional_groups['Ether'] += 1
                elif atomic_num == 16:  # Sulfur
                    atom_types['Sulfur'] += 1
                    functional_groups['Sulfur_containing'] += 1
                elif atomic_num == 15:  # Phosphorus
                    atom_types['Phosphorus'] += 1
                    functional_groups['Phosphate'] += 1
                elif atomic_num == 9:   # Fluorine
                    atom_types['Fluorine'] += 1
                    functional_groups['Fluorinated'] += 1
                elif atomic_num == 17:  # Chlorine
                    atom_types['Chlorine'] += 1
                    functional_groups['Chlorinated'] += 1
                elif atomic_num == 35:  # Bromine
                    atom_types['Bromine'] += 1
                    functional_groups['Brominated'] += 1
                elif atomic_num == 53:  # Iodine
                    atom_types['Iodine'] += 1
                    functional_groups['Iodinated'] += 1
            
            # Check for common pharmacophore patterns
            # Look for hydrogen bond donors/acceptors
            h_donors = 0
            h_acceptors = 0
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() in [7, 8] and atom.GetTotalNumHs() > 0:  # N, O with H
                    h_donors += 1
                if atom.GetAtomicNum() in [7, 8]:  # N, O can be acceptors
                    h_acceptors += 1
            
            functional_groups[f'H_donors_{h_donors}'] += 1
            functional_groups[f'H_acceptors_{h_acceptors}'] += 1
    
    return ring_systems, atom_types, functional_groups

def analyze_by_mechanism_of_action(molecules, names, csv_file):
    """
    Analyze clustering based on mechanism of action
    """
    # Load the CSV to get mechanism of action data
    df = pd.read_csv(csv_file)
    
    # Create a mapping from drug name to mechanism of action
    moa_mapping = {}
    for _, row in df.iterrows():
        drug_name = row['drug']
        moa_broad = row['moa-broad']
        moa_fine = row['moa-fine']
        moa_mapping[drug_name] = {'broad': moa_broad, 'fine': moa_fine}
    
    # Analyze distribution of MOAs in the loaded molecules
    moa_counts = Counter()
    for name in names:
        if name in moa_mapping:
            moa = moa_mapping[name]['broad']
            moa_counts[moa] += 1
    
    return moa_counts, moa_mapping

def main():
    print("Loading molecules from SDF files...")
    molecules, names = load_molecules_from_sdf('sdfs')
    print(f"Loaded {len(molecules)} molecules")
    
    # Filter out any None molecules
    valid_data = [(mol, name) for mol, name in zip(molecules, names) if mol is not None]
    molecules = [x[0] for x in valid_data]
    names = [x[1] for x in valid_data]
    print(f"Valid molecules after filtering: {len(molecules)}")
    
    # Find common pharmacophores and structural features
    print("\nAnalyzing pharmacophore features...")
    ring_systems, atom_types, functional_groups = find_common_pharmacophores(molecules, names)
    
    print("\nMost common ring systems:")
    for ring_type, count in ring_systems.most_common(10):
        print(f"{ring_type}: {count}")
    
    print("\nMost common atom types:")
    for atom_type, count in atom_types.most_common(10):
        print(f"{atom_type}: {count}")
    
    print("\nMost common functional groups/pharmacophores:")
    for func_group, count in functional_groups.most_common(15):
        print(f"{func_group}: {count}")
    
    # Analyze by mechanism of action
    print("\nAnalyzing by mechanism of action...")
    moa_counts, moa_mapping = analyze_by_mechanism_of_action(molecules, names, 'tahoe_100m_drug_metadata.csv')
    
    print("\nTop mechanisms of action:")
    for moa, count in moa_counts.most_common(10):
        print(f"{moa}: {count}")
    
    # Calculate molecular descriptors for each cluster
    print("\nCalculating molecular descriptors...")
    mol_descriptors = []
    for mol in molecules:
        try:
            weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            tpsa = Descriptors.TPSA(mol)
            
            mol_descriptors.append({
                'MW': weight,
                'LogP': logp,
                'RotatableBonds': rotatable_bonds,
                'HDonors': h_donors,
                'HAcceptors': h_acceptors,
                'TPSA': tpsa
            })
        except:
            # If calculation fails, add default values
            mol_descriptors.append({
                'MW': 0,
                'LogP': 0,
                'RotatableBonds': 0,
                'HDonors': 0,
                'HAcceptors': 0,
                'TPSA': 0
            })
    
    # Convert to DataFrame for easier analysis
    desc_df = pd.DataFrame(mol_descriptors)
    
    print("\nMolecular descriptor statistics:")
    print(desc_df.describe())
    
    # Calculate some summary statistics
    print(f"\nMolecular weight - Mean: {desc_df['MW'].mean():.2f}, Std: {desc_df['MW'].std():.2f}, Min: {desc_df['MW'].min():.2f}, Max: {desc_df['MW'].max():.2f}")
    print(f"LogP - Mean: {desc_df['LogP'].mean():.2f}, Std: {desc_df['LogP'].std():.2f}, Min: {desc_df['LogP'].min():.2f}, Max: {desc_df['LogP'].max():.2f}")
    
    # Perform clustering with ECFP fingerprints
    print("\nCalculating fingerprints for clustering...")
    fingerprints = calculate_fingerprints(molecules, fp_type='morgan', n_bits=2048)
    
    # Convert fingerprints to numpy array for clustering
    from rdkit import DataStructs
    fp_list = []
    for fp in fingerprints:
        arr = np.zeros((len(fp),))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
    
    fp_array = np.array(fp_list)
    
    # Perform Agglomerative clustering with different numbers of clusters
    print("Performing clustering analysis...")
    silhouette_scores = []
    cluster_range = range(2, min(50, len(molecules)//2))  # Limit number of clusters
    
    for n_clusters in cluster_range:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clustering.fit_predict(fp_array)
        silhouette_avg = silhouette_score(fp_array, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    if silhouette_scores:
        best_n_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"Best number of clusters based on silhouette score: {best_n_clusters}")
        
        # Perform final clustering with best number of clusters
        final_clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
        final_labels = final_clustering.fit_predict(fp_array)
        
        # Count molecules per cluster
        cluster_counts = Counter(final_labels)
        
        print(f"\nCluster distribution:")
        for cluster_id, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"Cluster {cluster_id}: {count} molecules")
    
    print("\nAnalysis complete! The results provide insights into:")
    print("- Common structural features and pharmacophores")
    print("- Mechanism of action distribution")
    print("- Molecular descriptor ranges (MW, LogP, etc.)")
    print("- Potential molecular clusters based on structural similarity")

if __name__ == "__main__":
    main()