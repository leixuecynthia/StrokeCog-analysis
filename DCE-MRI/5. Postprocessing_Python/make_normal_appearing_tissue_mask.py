#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Removes "lesions" from whole brain mask and visualises for QC.
"""
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up dark theme
plt.style.use('dark_background')

def create_binary_mask(data, threshold=0.8):
    """Create binary mask from data"""
    return (data > threshold).astype(np.int8)

def find_best_slice(mask_data):
    """Find the slice with the largest mask area"""
    mask_areas = [np.sum(mask_data[:, :, i] > 0) for i in range(mask_data.shape[2])]
    return np.argmax(mask_areas)

def create_visualization(t1_data, stroke_mask, perilesion_mask, wmh_mask, brain_mask, 
                        output_path, patient_id, stats, has_stroke, has_perilesion):
    
    # Find best slice based on combined ROI presence
    combined_mask = wmh_mask | brain_mask
    if has_stroke:
        combined_mask = combined_mask | stroke_mask
    if has_perilesion:
        combined_mask = combined_mask | perilesion_mask
    
    best_slice = find_best_slice(combined_mask)
    
    # Get surrounding slices
    slices_to_show = [
        max(0, best_slice - 1),
        best_slice,
        min(t1_data.shape[2] - 1, best_slice + 1)
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Plot each slice
    for idx, slice_num in enumerate(slices_to_show):
        # Get slice data
        t1_slice = np.rot90(t1_data[:, :, slice_num])
        wmh_slice = np.rot90(wmh_mask[:, :, slice_num])
        brain_slice = np.rot90(brain_mask[:, :, slice_num])
        
        if has_stroke:
            stroke_slice = np.rot90(stroke_mask[:, :, slice_num])
        if has_perilesion:
            perilesion_slice = np.rot90(perilesion_mask[:, :, slice_num])
        
        # Top row: T1 only
        vmin, vmax = np.percentile(t1_slice[t1_slice > 0], [1, 99]) if np.any(t1_slice > 0) else (0, 1)
        axes[0, idx].imshow(t1_slice, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0, idx].set_title(f'Slice {slice_num}: T1', fontsize=10)
        axes[0, idx].axis('off')
        
        # Bottom row: T1 with all ROIs overlaid
        axes[1, idx].imshow(t1_slice, cmap='gray', vmin=vmin, vmax=vmax)
        
        # Overlay ROIs in order: brain (bottom), WMH, perilesion, stroke (top)
        # Brain (purple) - lowest layer
        axes[1, idx].imshow(np.ma.masked_where(~brain_slice, brain_slice), 
                           cmap='Purples', alpha=0.3, vmin=0, vmax=1)
        
        # WMH (orange)
        axes[1, idx].imshow(np.ma.masked_where(~wmh_slice, wmh_slice), 
                           cmap='Oranges', alpha=0.5, vmin=0, vmax=1)
        
        # Perilesion (yellow) - only if present
        if has_perilesion:
            axes[1, idx].imshow(np.ma.masked_where(~perilesion_slice, perilesion_slice), 
                               cmap='YlOrBr', alpha=0.6, vmin=0, vmax=1)
        
        # Stroke (red) - only if present
        if has_stroke:
            axes[1, idx].imshow(np.ma.masked_where(~stroke_slice, stroke_slice), 
                               cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        
        axes[1, idx].set_title(f'Slice {slice_num}: All ROIs', fontsize=10)
        axes[1, idx].axis('off')
    
    # Create custom legend based on available ROIs
    legend_patches = []
    if has_stroke:
        legend_patches.append(mpatches.Patch(color='red', alpha=0.6, label='Stroke'))
    if has_perilesion:
        legend_patches.append(mpatches.Patch(color='yellow', alpha=0.6, label='Perilesion'))
    legend_patches.append(mpatches.Patch(color='orange', alpha=0.5, label='WMH'))
    legend_patches.append(mpatches.Patch(color='purple', alpha=0.3, label='Brain (no CSF, no lesions)'))
    
    fig.legend(handles=legend_patches, 
              loc='upper right', bbox_to_anchor=(0.98, 0.97), fontsize=9)
    
    # Add title with statistics
    title_parts = [f'Patient {patient_id} - All ROIs Check']
    roi_info = []
    if has_stroke:
        roi_info.append(f'Stroke: {stats["stroke_voxels"]}')
    if has_perilesion:
        roi_info.append(f'Perilesion: {stats["perilesion_voxels"]}')
    roi_info.append(f'WMH: {stats["wmh_voxels"]}')
    roi_info.append(f'Brain: {stats["brain_voxels"]} voxels')
    
    title_parts.append(' | '.join(roi_info))
    title_parts.append(f'Total overlap detected: {stats["total_overlap"]} voxels')
    
    title_text = '\n'.join(title_parts)
    
    plt.suptitle(title_text, y=0.98, size=11, weight='bold')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'all_rois_{patient_id}.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def process_patient(timepoint, patient_id, base_path, qc_folder):
    """Process a single patient"""
    
    patient_path = os.path.join(base_path, timepoint, patient_id)
    structural_path = os.path.join(patient_path, "structural")
    
    # Define file paths - stroke and perilesion are optional
    brain_input_path = os.path.join(structural_path, "brain_roi_noCSF_zeroed.nii")
    stroke_input_path = os.path.join(structural_path, "stroke_roi_zeroed.nii")
    wmh_input_path = os.path.join(structural_path, "wmh_zeroed.nii")
    perilesion_input_path = os.path.join(structural_path, "perilesion_2_zeroed.nii")
    t1_input_path = os.path.join(structural_path, "3D_T1w_zeroed.nii")
    
    brain_output_path = os.path.join(structural_path, "brain_roi_noCSF_nolesions_zeroed.nii")
    
    # Check if required files exist (brain, WMH, T1 are required)
    required_files = [brain_input_path, wmh_input_path, t1_input_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Skipping {patient_id}: Missing required files:")
        for f in missing_files:
            print(f"  - {os.path.basename(f)}")
        return None
    
    # Check if stroke and perilesion exist
    has_stroke = os.path.exists(stroke_input_path)
    has_perilesion = os.path.exists(perilesion_input_path)
    
    # Load all images
    print(f"Processing {patient_id}...")
    if not has_stroke:
        print(f"  Note: No stroke lesion for {patient_id}")
    if not has_perilesion:
        print(f"  Note: No perilesion for {patient_id}")
    
    brain_img = nib.load(brain_input_path)
    wmh_img = nib.load(wmh_input_path)
    t1_img = nib.load(t1_input_path)
    
    # Get data
    brain_data = brain_img.get_fdata()
    wmh_data = wmh_img.get_fdata()
    t1_data = t1_img.get_fdata()
    
    # Create binary masks
    brain_mask = create_binary_mask(brain_data)
    wmh_mask = create_binary_mask(wmh_data)
    
    # Initialize stroke and perilesion masks
    stroke_mask = None
    perilesion_mask = None
    stroke_voxels = 0
    perilesion_voxels = 0
    
    if has_stroke:
        stroke_img = nib.load(stroke_input_path)
        stroke_data = stroke_img.get_fdata()
        stroke_mask = create_binary_mask(stroke_data)
        stroke_voxels = np.sum(stroke_mask)
    else:
        stroke_mask = np.zeros_like(brain_mask)
    
    if has_perilesion:
        perilesion_img = nib.load(perilesion_input_path)
        perilesion_data = perilesion_img.get_fdata()
        perilesion_mask = create_binary_mask(perilesion_data)
        perilesion_voxels = np.sum(perilesion_mask)
    else:
        perilesion_mask = np.zeros_like(brain_mask)
    
    # Store original voxel counts
    original_brain_voxels = np.sum(brain_mask)
    wmh_voxels = np.sum(wmh_mask)
    
    print(f"Original brain (no CSF): {original_brain_voxels} voxels")
    if has_stroke:
        print(f"Stroke: {stroke_voxels} voxels")
    if has_perilesion:
        print(f"Perilesion: {perilesion_voxels} voxels")
    print(f"WMH: {wmh_voxels} voxels")
    
    # Create brain mask by subtracting all lesion ROIs
    brain_nolesions_mask = brain_mask.copy()
    brain_nolesions_mask[stroke_mask == 1] = 0
    brain_nolesions_mask[wmh_mask == 1] = 0
    brain_nolesions_mask[perilesion_mask == 1] = 0
    
    final_brain_voxels = np.sum(brain_nolesions_mask)
    subtracted_voxels = original_brain_voxels - final_brain_voxels
    
    print(f"Final brain (no CSF, no lesions): {final_brain_voxels} voxels")
    print(f"Voxels removed: {subtracted_voxels}")
    
    # Check for overlaps between all ROI pairs
    overlaps = {}
    
    # Always check WMH vs brain
    overlaps['wmh_brain'] = np.sum((wmh_mask == 1) & (brain_nolesions_mask == 1))
    
    # Check stroke overlaps only if stroke exists
    if has_stroke:
        overlaps['stroke_wmh'] = np.sum((stroke_mask == 1) & (wmh_mask == 1))
        overlaps['stroke_brain'] = np.sum((stroke_mask == 1) & (brain_nolesions_mask == 1))
    
    # Check perilesion overlaps only if perilesion exists
    if has_perilesion:
        overlaps['perilesion_wmh'] = np.sum((perilesion_mask == 1) & (wmh_mask == 1))
        overlaps['perilesion_brain'] = np.sum((perilesion_mask == 1) & (brain_nolesions_mask == 1))
    
    # Check stroke-perilesion overlap only if both exist
    if has_stroke and has_perilesion:
        overlaps['stroke_perilesion'] = np.sum((stroke_mask == 1) & (perilesion_mask == 1))
    
    total_overlap = sum(overlaps.values())
    
    # Report overlaps
    print("Overlap checks:")
    for pair, count in overlaps.items():
        if count > 0:
            print(f"WARNING: {pair} overlap = {count} voxels")
        else:
            print(f"{pair} overlap = {count} voxels ✓")
    
    # Assert no overlaps
    assert total_overlap == 0, f"ERROR: Found {total_overlap} total overlapping voxels in {patient_id}!"
    
    # Save the brain_nolesions mask
    nib.save(nib.Nifti1Image(brain_nolesions_mask, brain_img.affine, brain_img.header), 
             brain_output_path)
    print("Saved: brain_roi_noCSF_nolesions_zeroed.nii")
    
    # Prepare statistics for visualization
    stats = {
        'stroke_voxels': int(stroke_voxels),
        'perilesion_voxels': int(perilesion_voxels),
        'wmh_voxels': int(wmh_voxels),
        'brain_voxels': int(final_brain_voxels),
        'total_overlap': int(total_overlap)
    }
    
    # Create visualization
    create_visualization(t1_data, stroke_mask, perilesion_mask, wmh_mask, 
                        brain_nolesions_mask, qc_folder, patient_id, stats,
                        has_stroke, has_perilesion)
    print("Saved: QC image")
    
    # Return results
    results = {
        'patient_id': patient_id,
        'has_stroke': has_stroke,
        'has_perilesion': has_perilesion,
        'original_brain_voxels': int(original_brain_voxels),
        'final_brain_voxels': int(final_brain_voxels),
        'stroke_voxels': int(stroke_voxels),
        'wmh_voxels': int(wmh_voxels),
        'perilesion_voxels': int(perilesion_voxels),
        'voxels_subtracted': int(subtracted_voxels),
        'total_overlap': int(total_overlap),
    }
    
    # Add overlap details
    for k, v in overlaps.items():
        results[f'overlap_{k}'] = int(v)
    
    return results

# Base path
base_path = "/Volumes/G-DRIVE/Stroke_Impact/Manchester" # Data directory
timepoint = "6months"

# Create QC output folder
qc_folder = os.path.join(base_path, "QC_images", "check_all_rois_PDGFB") # Quality control folder
os.makedirs(qc_folder, exist_ok=True)

# Read patient IDs
id_file = os.path.join(base_path, "6month_IDs.txt") # Path to participant ID file, text file in data directory.
with open(id_file, "r") as f:
    patient_ids = [line.strip() for line in f]

print(f"Processing {len(patient_ids)} patients...\n")

# Process all patients
all_results = []
for patient_id in patient_ids:
    try:
        results = process_patient(timepoint, patient_id, base_path, qc_folder)
        if results:
            all_results.append(results)
        print(f"Successfully processed {patient_id}\n")
    except AssertionError as e:
        print(f"OVERLAP ERROR for {patient_id}: {str(e)}\n")
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}\n")
        import traceback
        print(traceback.format_exc())

# Save summary to excel
if all_results:
    import pandas as pd
    df = pd.DataFrame(all_results)
    summary_csv = os.path.join(qc_folder, "roi_processing_summary.csv") # Output filename
    df.to_csv(summary_csv, index=False)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_csv}")
    print(f"Successfully processed: {len(all_results)}/{len(patient_ids)} patients")
    print(f"Patients with stroke: {df['has_stroke'].sum()}")
    print(f"Patients with perilesion: {df['has_perilesion'].sum()}")
    print(f"Patients with overlaps: {(df['total_overlap'] > 0).sum()}")
    print(f"QC images saved to: {qc_folder}")
    print(f"{'='*60}")