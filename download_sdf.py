import pandas as pd
import requests
import os
from pathlib import Path

def download_sdf_files(csv_file_path):
    """
    Download SDF files from PubChem using CIDs from the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Create the sdfs directory if it doesn't exist
    sdfs_dir = Path("sdfs")
    sdfs_dir.mkdir(exist_ok=True)
    
    # Iterate through the dataframe to download SDF files
    for index, row in df.iterrows():
        drug_name = row['drug']
        cid = row['pubchem_cid']
        
        # Check if CID is valid (not NaN or null)
        if pd.isna(cid) or cid == '':
            print(f"Skipping {drug_name} - no valid CID")
            continue
        
        # Convert CID to integer if it's not NaN
        try:
            cid_int = int(float(cid))
        except (ValueError, TypeError):
            print(f"Skipping {drug_name} - invalid CID: {cid}")
            continue
        
        # Create a safe filename by replacing problematic characters
        safe_drug_name = "".join(c for c in drug_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        filename = f"{safe_drug_name}_{cid_int}.sdf"
        filepath = sdfs_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            print(f"Skipping {drug_name} - file already exists")
            continue
        
        # Download the SDF file from PubChem
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_int}/SDF"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded SDF for {drug_name} (CID: {cid_int})")
            else:
                print(f"Failed to download SDF for {drug_name} (CID: {cid_int}) - Status code: {response.status_code}")
        except Exception as e:
            print(f"Error downloading SDF for {drug_name} (CID: {cid_int}): {str(e)}")

if __name__ == "__main__":
    csv_file_path = "tahoe_100m_drug_metadata.csv"
    download_sdf_files(csv_file_path)