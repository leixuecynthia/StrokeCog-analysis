# -*- coding: utf-8 -*-
"""
Dilates stroke roi to generate new "peri-lesion" roi.
"""
import os
import nibabel as nib
import numpy as np
from scipy import ndimage

def create_perilesion_roi(stroke_roi_path, brain_mask_path, perilesion_path, dilation_mm=10):
    """
    Create a ring ROI around a stroke region by dilating the stroke ROI,
    excluding CSF and non-brain tissue.

    -----------
    stroke_roi_path : str
        Path to the stroke ROI NIfTI file
    brain_mask_path : str
        Path to the brain mask NIfTI file (brain_roi_noCSF.nii)
    perilesion_path : str
        Path to save the perilesion ROI NIfTI file
    dilation_mm : int
        Dilation size in mm (default: 10mm = 1cm)
    """
    # Load the stroke ROI image
    stroke_img = nib.load(stroke_roi_path)
    stroke_data = stroke_img.get_fdata()
    
    # Load the brain mask (no CSF)
    brain_mask_img = nib.load(brain_mask_path)
    brain_mask_data = brain_mask_img.get_fdata()
    
    # Get voxel dimensions to adjust dilation size
    voxel_dims = stroke_img.header.get_zooms()
    
    # Calculate dilation iterations for each dimension based on voxel size
    # For 1mm³ voxels, this would be 10 iterations for 1cm
    dilation_iterations = [int(round(dilation_mm / dim)) for dim in voxel_dims[:3]]
    
    # Create a spherical structuring element for dilation
    struct_elem = ndimage.generate_binary_structure(3, 2)
    
    # Dilate the stroke ROI
    dilated_roi = ndimage.binary_dilation(
        stroke_data > 0, 
        structure=struct_elem, 
        iterations=max(dilation_iterations)
    ).astype(np.int16)
    
    # Create the ring ROI by subtracting the original stroke ROI
    ring_roi = dilated_roi - (stroke_data > 0)
    
    # Mask out CSF and non-brain tissue
    # Only keep voxels that are in the brain mask
    ring_roi_masked = ring_roi * (brain_mask_data > 0)
    
    # Create a new NIfTI image with the masked ring ROI
    perilesion_img = nib.Nifti1Image(ring_roi_masked, stroke_img.affine, stroke_img.header)
    
    # Save the perilesion ROI
    nib.save(perilesion_img, perilesion_path)
    
    return perilesion_path


# Base directory
base_dir = r"/Volumes/G-DRIVE/Stroke_Impact/Manchester" # Data directory

# Read patient IDs from file
with open(os.path.join(base_dir, "6month_IDs.txt"), "r") as f: # Define participant ID file path
    patient_ids = [line.strip() for line in f if line.strip()]

# Process each patient
for patient_id in patient_ids:
    # Construct paths
    patient_dir = os.path.join(base_dir, "6months", patient_id, "structural")
    stroke_roi_path = os.path.join(patient_dir, "stroke_roi_zeroed.nii")
    brain_mask_path = os.path.join(patient_dir, "fs_brain_roi_noCSF_zeroed.nii")
    perilesion_path = os.path.join(patient_dir, "perilesion_2_zeroed.nii")
    
    # Check if required files exist
    if os.path.exists(stroke_roi_path) and os.path.exists(brain_mask_path):
        try:
            print(f"Processing patient {patient_id}...")
            create_perilesion_roi(stroke_roi_path, brain_mask_path, perilesion_path)
            print(f"Created perilesion ROI for patient {patient_id}")
        except Exception as e:
            print(f"Error processing patient {patient_id}: {str(e)}")
    else:
        if not os.path.exists(stroke_roi_path):
            print(f"Stroke ROI not found for patient {patient_id}")
        if not os.path.exists(brain_mask_path):
            print(f"Brain mask not found for patient {patient_id}")