import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.ML.Cluster import Butina
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os

def create_results_folder():
    """Create results directory if it doesn't exist"""
    results_dir = Path("analysis_results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

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

def find_optimal_clusters(fingerprints, method='agglomerative', max_clusters=50):
    """
    Find optimal number of clusters using silhouette analysis
    """
    from rdkit import DataStructs
    
    # Convert fingerprints to numpy array
    fp_list = []
    for fp in fingerprints:
        arr = np.zeros((len(fp),))
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr)
    
    fp_array = np.array(fp_list)
    
    # Test different numbers of clusters
    cluster_range = range(3, min(max_clusters, len(fingerprints)//2))  # At least 3, up to reasonable max
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        if method == 'agglomerative':
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        cluster_labels = clustering.fit_predict(fp_array)
        try:
            silhouette_avg = silhouette_score(fp_array, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        except:
            silhouette_scores.append(-1)  # In case of failure
    
    if silhouette_scores:
        best_idx = np.argmax(silhouette_scores)
        best_n_clusters = cluster_range[best_idx]
        best_score = silhouette_scores[best_idx]
        
        print(f"Best number of clusters: {best_n_clusters} with silhouette score: {best_score:.3f}")
        
        # Perform final clustering with best number of clusters
        if method == 'agglomerative':
            final_clustering = AgglomerativeClustering(n_clusters=best_n_clusters)
        else:
            final_clustering = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
        
        final_labels = final_clustering.fit_predict(fp_array)
        
        return final_labels, best_n_clusters, best_score
    else:
        # Fallback: just create 10 clusters
        clustering = AgglomerativeClustering(n_clusters=10)
        labels = clustering.fit_predict(fp_array)
        return labels, 10, 0.0

def save_clustering_results(results_dir, names, cluster_labels, moa_mapping, molecules):
    """
    Save clustering results to files in the results directory
    """
    # Create cluster assignments dataframe
    cluster_df = pd.DataFrame({
        'Drug_Name': names,
        'Cluster_ID': cluster_labels
    })
    
    # Add mechanism of action info if available
    moa_broad = []
    moa_fine = []
    for name in names:
        if name in moa_mapping:
            moa_broad.append(moa_mapping[name]['broad'])
            moa_fine.append(moa_mapping[name]['fine'])
        else:
            moa_broad.append('Unknown')
            moa_fine.append('Unknown')
    
    cluster_df['MOA_Broad'] = moa_broad
    cluster_df['MOA_Fine'] = moa_fine
    
    # Calculate molecular descriptors for each molecule
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
    
    # Add descriptors to the dataframe
    desc_df = pd.DataFrame(mol_descriptors)
    for col in desc_df.columns:
        cluster_df[f'Mol_{col}'] = desc_df[col]
    
    # Save clustering results
    cluster_df.to_csv(results_dir / 'clustering_results.csv', index=False)
    
    # Create a summary of each cluster
    cluster_summary = cluster_df.groupby('Cluster_ID').agg({
        'Drug_Name': 'count',
        'MOA_Broad': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
        'Mol_MW': ['mean', 'std'],
        'Mol_LogP': ['mean', 'std'],
        'Mol_HAcceptors': ['mean', 'std']
    }).round(2)
    
    # Flatten column names
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    cluster_summary = cluster_summary.rename(columns={'Drug_Name_count': 'Drug_Count'})
    cluster_summary.to_csv(results_dir / 'cluster_summary.csv')
    
    # Save cluster composition
    cluster_composition = {}
    for cluster_id in sorted(set(cluster_labels)):
        drugs_in_cluster = cluster_df[cluster_df['Cluster_ID'] == cluster_id]['Drug_Name'].tolist()
        moas_in_cluster = cluster_df[cluster_df['Cluster_ID'] == cluster_id]['MOA_Broad'].tolist()
        cluster_composition[cluster_id] = {
            'drugs': drugs_in_cluster,
            'moas': moas_in_cluster,
            'count': len(drugs_in_cluster)
        }
    
    # Write detailed cluster info to text file
    with open(results_dir / 'cluster_details.txt', 'w') as f:
        f.write("Molecular Clustering Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total number of drugs: {len(names)}\n")
        f.write(f"Total number of clusters: {len(set(cluster_labels))}\n\n")
        
        for cluster_id in sorted(cluster_composition.keys()):
            f.write(f"Cluster {cluster_id}:\n")
            f.write(f"  Drug Count: {cluster_composition[cluster_id]['count']}\n")
            f.write(f"  Most Common MOA: {Counter(cluster_composition[cluster_id]['moas']).most_common(1)[0][0]}\n")
            f.write(f"  Drugs: {', '.join(cluster_composition[cluster_id]['drugs'][:10])}{'...' if len(cluster_composition[cluster_id]['drugs']) > 10 else ''}\n\n")
    
    print(f"Results saved to {results_dir}/")

def main():
    results_dir = create_results_folder()
    
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
    
    # Save pharmacophore analysis
    with open(results_dir / 'pharmacophore_analysis.txt', 'w') as f:
        f.write("Pharmacophore and Structural Feature Analysis\n")
        f.write("=" * 50 + "\n")
        
        f.write("\nMost common ring systems:\n")
        for ring_type, count in ring_systems.most_common(20):
            f.write(f"{ring_type}: {count}\n")
        
        f.write("\nMost common atom types:\n")
        for atom_type, count in atom_types.most_common(20):
            f.write(f"{atom_type}: {count}\n")
        
        f.write("\nMost common functional groups/pharmacophores:\n")
        for func_group, count in functional_groups.most_common(20):
            f.write(f"{func_group}: {count}\n")
    
    print("Ring systems, atom types, and functional groups analysis completed.")
    
    # Analyze by mechanism of action
    print("\nAnalyzing by mechanism of action...")
    moa_counts, moa_mapping = analyze_by_mechanism_of_action(molecules, names, 'tahoe_100m_drug_metadata.csv')
    
    with open(results_dir / 'moa_analysis.txt', 'w') as f:
        f.write("Mechanism of Action Analysis\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total unique MOAs: {len(moa_counts)}\n\n")
        
        f.write("Top mechanisms of action:\n")
        for moa, count in moa_counts.most_common(20):
            f.write(f"{moa}: {count}\n")
    
    print("MOA analysis completed.")
    
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
    
    # Save molecular descriptors summary
    with open(results_dir / 'molecular_descriptors.txt', 'w') as f:
        f.write("Molecular Descriptor Statistics\n")
        f.write("=" * 30 + "\n")
        f.write(str(desc_df.describe()) + "\n\n")
        
        f.write(f"Molecular weight - Mean: {desc_df['MW'].mean():.2f}, Std: {desc_df['MW'].std():.2f}, Min: {desc_df['MW'].min():.2f}, Max: {desc_df['MW'].max():.2f}\n")
        f.write(f"LogP - Mean: {desc_df['LogP'].mean():.2f}, Std: {desc_df['LogP'].std():.2f}, Min: {desc_df['LogP'].min():.2f}, Max: {desc_df['LogP'].max():.2f}\n")
        f.write(f"Hydrogen Bond Acceptors - Mean: {desc_df['HAcceptors'].mean():.2f}\n")
    
    print("Molecular descriptor analysis completed.")
    
    # Perform clustering with ECFP fingerprints
    print("\nCalculating fingerprints for clustering...")
    fingerprints = calculate_fingerprints(molecules, fp_type='morgan', n_bits=2048)
    
    # Find optimal number of clusters
    print("Performing clustering analysis to find optimal number of clusters...")
    cluster_labels, n_clusters, silhouette_score_val = find_optimal_clusters(fingerprints, method='agglomerative', max_clusters=20)
    
    print(f"Clustering completed with {n_clusters} clusters and silhouette score: {silhouette_score_val:.3f}")
    
    # Count molecules per cluster
    cluster_counts = Counter(cluster_labels)
    
    print(f"\nCluster distribution:")
    for cluster_id, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"Cluster {cluster_id}: {count} molecules")
    
    # Save all results
    print("\nSaving results...")
    save_clustering_results(results_dir, names, cluster_labels, moa_mapping, molecules)
    
    print("\nAnalysis complete! Results saved in 'analysis_results' folder:")
    print("- clustering_results.csv: Complete clustering assignments")
    print("- cluster_summary.csv: Summary statistics for each cluster")
    print("- cluster_details.txt: Detailed information about each cluster")
    print("- pharmacophore_analysis.txt: Analysis of common structural features")
    print("- moa_analysis.txt: Mechanism of action distribution")
    print("- molecular_descriptors.txt: Molecular property statistics")

if __name__ == "__main__":
    main()