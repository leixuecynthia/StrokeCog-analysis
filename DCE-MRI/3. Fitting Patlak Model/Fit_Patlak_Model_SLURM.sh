#!/bin/bash --login
# Script for fitting Patlak model of tracer leakage to DCE-MRI data on a voxel-wise basis using Madym. 
#SBATCH -p multicore
#SBATCH -c 8
#SBATCH -a 1-60
#SBATCH -t 72:00:00
#SBATCH -o %x_%A_%a.out
#SBATCH -J NEW_IDs_Aug25_cut60

here=/mnt/iusers01/fatpou01/bmh01/p15094oj # Data directory

ID=`awk "NR==$SLURM_ARRAY_TASK_ID"  $here/ModelSelection2/6month_IDs.txt` # Patient ID list, text file
Hct=`awk "NR==$SLURM_ARRAY_TASK_ID" $here/ModelSelection2/6month_IDs.txt`

cd $SLURM_SUBMIT_DIR

module load apps/singularity/madym/4.23.0

madym "madym_DCE -m PATLAK -o $here/scratch/Stroke_Impact/Manchester/6months/${ID}/dce_fitting/Patlak -d rdyn_ --dyn_dir $here/scratch/Stroke_Impact/Manchester/6months/${ID}/dynamic_series --sequence_format %01u --sequence_start 2 --sequence_step 1 --n_dyns 156 --Ct 0 --T1 $here/scratch/Stroke_Impact/Manchester/6months/${ID}/t1_map/T1.nii --r1 3.400000e+00 -D 1.000000e-01 --M0_ratio 1 --no_opt 0 --Ct_sig 1 --Ct_mod 1 --test_enh 1 --max_iter 1000 --opt_type BLEIC --overwrite 1 -H ${Hct} -i 8 --aif $here/scratch/Stroke_Impact/Manchester/6months/${ID}/aif/AIFmap.txt --audit_dir $here/scratch/Stroke_Impact/Manchester/6months/${ID}/madym_files/logs --init_params 1.000000e-03,2.000000e-02,0 --upper_bounds 10,1,20 --lower_bounds -10,0,-20"

madym "madym_DCE -m PATLAK -o $here/scratch/Stroke_Impact/Manchester/6months/${ID}/dce_fitting/Patlak_cut60 -d Ct_sig --dyn_dir $here/scratch/Stroke_Impact/Manchester/6months/${ID}/dce_fitting/Patlak/Ct_sig/ --sequence_format %01u --sequence_start 2 --sequence_step 1 --n_dyns 156 --first 16 --Ct 1 --r1 3.400000e+00 -D 1.000000e-01 --no_opt 0 --Ct_sig 0 --Ct_mod 1 --test_enh 1 --max_iter 1000 --opt_type BLEIC --overwrite 1 -H ${Hct} -i 8  --aif $here/scratch/Stroke_Impact/Manchester/6months/${ID}/aif/AIFmap.txt --audit_dir $here/scratch/Stroke_Impact/Manchester/6months/${ID}/madym_files/logs --init_params 1.000000e-03,2.000000e-02,0 --upper_bounds 10,1,20 --lower_bounds -10,0,-20 --init_maps $here/scratch/Stroke_Impact/Manchester/6months/${ID}/dce_fitting/Patlak --init_map_params 3 --fixed_params 3 --fixed_values 0"
