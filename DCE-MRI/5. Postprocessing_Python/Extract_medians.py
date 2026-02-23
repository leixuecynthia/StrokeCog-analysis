# -*- coding: utf-8 -*-
"""
Extract median Ktrans/vp from maps and roi .nii files. 
"""

import os
import nibabel as nib
import numpy as np
import pandas as pd

median_model = "Patlak"  # Define tracer kinetic model used
cut = 'cut60'

def extract_median_parameters(ktrans_file, vp_file, mask_file):
    # Load the parameter maps
    ktrans_img = nib.load(ktrans_file)
    ktrans_data = ktrans_img.get_fdata()
    
    vp_img = nib.load(vp_file)
    vp_data = vp_img.get_fdata()
    
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()
    
    # Apply mask and vp threshold
    mask_indices = (mask_data > 0.9) & (vp_data >= 0.0001)
    
    masked_ktrans = ktrans_data[mask_indices]
    masked_vp = vp_data[mask_indices]
    
    # Remove NaN values
    valid_indices = ~np.isnan(masked_ktrans) & ~np.isnan(masked_vp)
    masked_ktrans = masked_ktrans[valid_indices]
    masked_vp = masked_vp[valid_indices]
    
    results = {}
    
    # Calculate medians if there are valid values
    if masked_ktrans.size > 0:
        results['ktrans'] = np.median(masked_ktrans)
    else:
        print(f"Warning: No valid Ktrans values in the mask for {ktrans_file}")
        results['ktrans'] = np.nan
        
    if masked_vp.size > 0:
        results['vp'] = np.median(masked_vp)
    else:
        print(f"Warning: No valid vp values in the mask for {vp_file}")
        results['vp'] = np.nan
        
    return results

def process_patient(patient_id, timepoint):
    base_dir = f"/Volumes/G-DRIVE/Stroke_Impact/Manchester/{timepoint}/{patient_id}/" # Define data directory
    structural_dir = f"/Volumes/G-DRIVE/Stroke_Impact/Manchester/{timepoint}/{patient_id}/structural/"
    ktrans_file = os.path.join(base_dir, "dce_model_selection_" + cut, "rKtrans_" + median_model + ".nii")
    vp_file = os.path.join(base_dir, "dce_model_selection_" + cut, "rv_p_" + median_model + ".nii")
    
    if not os.path.exists(ktrans_file) or not os.path.exists(vp_file):
        print(f"One or more required files not found for patient {patient_id}. Skipping.")
        return None
    
    regions = {
        "wb_nolesions": "brain_roi_noCSF_nolesions_zeroed.nii",
        "whole_brain": "brain_roi_noCSF_zeroed.nii",
        "wmh": "wmh_zeroed.nii",
        "stroke_roi": "stroke_roi_zeroed.nii",
        "perilesion_corr": "perilesion_2_zeroed.nii"
    }
    
    ktrans_results = {"Patient_ID": patient_id}
    vp_results = {"Patient_ID": patient_id}
    model_results = {"Patient_ID": patient_id}
    
    for region, mask_file in regions.items():
        mask_path = os.path.join(structural_dir, mask_file)
        if os.path.exists(mask_path):
            # Get parameter medians
            parameters = extract_median_parameters(ktrans_file, vp_file, mask_path)
            ktrans_results[region] = parameters['ktrans']
            vp_results[region] = parameters['vp']

        else:
            print(f"Mask file {mask_file} not found for patient {patient_id}.")
            ktrans_results[region] = None
            vp_results[region] = None
    
    return ktrans_results, vp_results, model_results

# Read patient IDs from the text file
timepoint = "6months"
sheet_name = "6month_IDs" # Sheet name in output Excel

patient_ids_file = "/Volumes/G-DRIVE/Stroke_Impact/Manchester/6month_IDs.txt" # Path to ID file
with open(patient_ids_file, 'r') as f:
    patient_ids = [line.strip() for line in f if line.strip()]

# Process each patient and collect results
ktrans_results = []
vp_results = []

for patient_id in patient_ids:
    print(f"Processing patient: {patient_id}")
    result = process_patient(patient_id, timepoint)
    if result:
        ktrans_results.append(result[0])
        vp_results.append(result[1])

# Create DataFrames from the results
ktrans_df = pd.DataFrame(ktrans_results)
vp_df = pd.DataFrame(vp_results)

# Define the output Excel file paths
ktrans_output_file = "/Volumes/G-DRIVE/Stroke_Impact/Manchester/" + timepoint + "/Ktrans_" + median_model + cut + "_median_values.xlsx"
vp_output_file = "/Volumes/G-DRIVE/Stroke_Impact/Manchester/" + timepoint + "/vp_" + median_model + cut + "_median_values.xlsx"

# Function to save DataFrame to Excel
def save_to_excel(df, output_file, sheet_name):
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
            if sheet_name in writer.book.sheetnames:
                idx = writer.book.sheetnames.index(sheet_name)
                writer.book.remove(writer.book.worksheets[idx])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

# Save all results
save_to_excel(ktrans_df, ktrans_output_file, sheet_name)
save_to_excel(vp_df, vp_output_file, sheet_name)

print(f"Ktrans results saved to {ktrans_output_file} in sheet '{sheet_name}'")
print(f"vp results saved to {vp_output_file} in sheet '{sheet_name}'")