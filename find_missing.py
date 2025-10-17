import pandas as pd
import os
from pathlib import Path

def find_missing_sdf_files(csv_file_path, sdfs_dir_path):
    """
    Find rows in the CSV that don't have corresponding SDF files
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Get list of existing SDF files
    sdfs_dir = Path(sdfs_dir_path)
    existing_sdf_files = set()
    
    for sdf_file in sdfs_dir.glob("*.sdf"):
        # Extract drug name from filename (before the last underscore and CID)
        filename = sdf_file.stem  # Remove .sdf extension
        # Split by last underscore to separate drug name from CID
        parts = filename.rsplit('_', 1)
        if len(parts) == 2:
            drug_name = parts[0]
            existing_sdf_files.add(drug_name)
    
    # Find missing entries
    missing_entries = []
    for index, row in df.iterrows():
        drug_name = row['drug']
        cid = row['pubchem_cid']
        
        # Create a safe filename similar to what the download script used
        safe_drug_name = "".join(c for c in drug_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        
        # Check if this drug name exists in our SDF files
        if safe_drug_name not in existing_sdf_files:
            missing_entries.append({
                'index': index,
                'drug': drug_name,
                'pubchem_cid': cid,
                'moa_broad': row['moa-broad'],
                'moa_fine': row['moa-fine'],
                'human_approved': row['human-approved'],
                'clinical_trials': row['clinical-trials']
            })
    
    return missing_entries

if __name__ == "__main__":
    csv_file_path = "tahoe_100m_drug_metadata.csv"
    sdfs_dir_path = "sdfs"
    
    missing_entries = find_missing_sdf_files(csv_file_path, sdfs_dir_path)
    
    print(f"Total missing entries: {len(missing_entries)}")
    print("\nMissing entries:")
    print("Index\tDrug Name\t\t\tCID\t\tMOA-Broad\t\tMOA-Fine\t\tApproved\tTrials")
    print("-" * 150)
    
    for entry in missing_entries:
        print(f"{entry['index']}\t{entry['drug'][:30]:<25}\t{entry['pubchem_cid']}\t\t"
              f"{entry['moa_broad'][:20]:<18}\t{entry['moa_fine'][:20]:<18}\t"
              f"{entry['human_approved']}\t\t{entry['clinical_trials']}")