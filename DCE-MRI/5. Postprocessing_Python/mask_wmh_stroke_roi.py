# -*- coding: utf-8 -*-
"""
Mask the WMH mask and make sure stroke roi is removed.
"""

import os
import nibabel as nib
import numpy as np

def create_binary_mask(data, threshold):
    return (data > threshold).astype(np.int8)

def process_patient(timepoint,patient_id):
    base_path = f"/Volumes/G-DRIVE/Stroke_Impact/Manchester/{timepoint}/{patient_id}"
    structural_path = os.path.join(base_path, "structural")
    
    # Process acute DWI infarct ROI
    dwi_input_path = os.path.join(base_path, "roi", "rAcute_DWI_Infarct_ROI_fixed.nii") # Input DWI path, registered to T1w space.
    stroke_roi_output_path = os.path.join(base_path, "roi", "rAcute_DWI_Infarct_ROI_fixed_cut.nii") # Input Stroke ROI drawn on DWI, also registered to DWI space.
    
    stroke_roi_mask = None
    if os.path.exists(dwi_input_path):
        dwi_img = nib.load(dwi_input_path)
        dwi_data = dwi_img.get_fdata()
        os.remove(stroke_roi_output_path)
        stroke_roi_mask = create_binary_mask(dwi_data, 0.8)  # Assuming any non-zero value is part of the ROI
        nib.save(nib.Nifti1Image(stroke_roi_mask, dwi_img.affine, dwi_img.header), stroke_roi_output_path)
        print(f"Saved stroke ROI mask for {patient_id}")
    else:
        print(f"No acute DWI infarct ROI found for {patient_id}")
    
    # Process WMH mask
    wmh_input_path = os.path.join(structural_path, "ples_lpa_mrFLAIR.nii")
    wmh_output_path = os.path.join(structural_path, "wmh.nii")
    
    wmh_img = nib.load(wmh_input_path)
    wmh_data = wmh_img.get_fdata()
    wmh_mask = create_binary_mask(wmh_data, 0.3)
    
    # Remove stroke ROI from WMH mask if it exists
    if stroke_roi_mask is not None:
        wmh_mask[stroke_roi_mask == 1] = 0
        # OVerlap checker
        overlap_check = np.sum((wmh_mask == 1) & (stroke_roi_mask == 1))
        print(f"Overlap voxels after subtraction for {patient_id}: {overlap_check}")
        assert overlap_check == 0, f"ERROR: Found {overlap_check} overlapping voxels!"
    
    # Save the final WMH mask
    nib.save(nib.Nifti1Image(wmh_mask, wmh_img.affine, wmh_img.header), wmh_output_path)
    print(f"Saved WMH mask for {patient_id}")
    

# Read patient IDs from the text file
timepoint = '6months'
with open("/Volumes/G-DRIVE/Stroke_Impact/Manchester/6month_IDs.txt", "r") as f:
    patient_ids = [line.strip() for line in f]
# Process each patient
for patient_id in patient_ids:
    try:
        process_patient(timepoint,patient_id)
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
