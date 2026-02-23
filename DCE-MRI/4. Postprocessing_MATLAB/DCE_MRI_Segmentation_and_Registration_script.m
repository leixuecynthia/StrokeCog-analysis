%% Script to segment structural images using SPM12 with LST toolbox. Registers parametric maps to 3D-T1w image space.

addpath('/Users/user/Documents/MATLAB/IMPaCT_fitting/Manchester') % Code directory
addpath('/Users/user/Documents/MATLAB/spm12')
addpath('/Users/user/Documents/MATLAB/fm_toolbox')

id_file = fileread(['/Volumes/G-DRIVE/Stroke_Impact/Manchester/6month_IDs.txt']); % Patient ID file, text file in data directory
id_list = strsplit(id_file);

%% Structural image segmentation: GM, WM, CSF, WMH
    for person=1:length(id_list)
        id = char(id_list(person))
        cd(['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id])
        mkdir('structural')
        if isfile('raw/WIP_3DT1_1mm_ISO_CS_2m23s.nii')
            copyfile('raw/WIP_3DT1_1mm_ISO_CS_2m23s.nii', 'structural/3D_T1w.nii')
            copyfile('raw/WIP_cs10_3D_Brain_VIEW_FLAIR_32chSHC.nii', 'structural/FLAIR.nii')
        elseif isfile('raw/3DT1_1mm_ISO_CS_2m23s.nii')
            copyfile('raw/3DT1_1mm_ISO_CS_2m23s.nii', 'structural/3D_T1w.nii')
            copyfile('raw/cs10_3D_Brain_VIEW_FLAIR_32chSHC.nii', 'structural/FLAIR.nii')
        else 
            continue
        end
        t1dir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/structural/']
        segment([t1dir '3D_T1w.nii']) % Segment T1w into GM, WM, CSF
        wmh_lpa([t1dir 'FLAIR.nii'],[t1dir '3D_T1w.nii']) % WML segmentation. Requires LST toolbox.
    end

%% Register everything to T1w space
for person=1:length(id_list)
        id = char(id_list(person))
        t1dir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/structural/'];
        dcedir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/dce_fitting/'];
        dyndir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/dynamic_series/'];
        msdir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/dce_model_selection_' cut '/'];

        dyn1 = double(load_nii([dyndir 'rdyn_1.nii']).img);
        dyn1_nii = make_nii(dyn1, [1.5 1.5 2]);
        save_nii(dyn1_nii, [t1dir 'rdyn_1.nii']);
        centre_header_file([t1dir 'rdyn_1.nii']);
        copyfile([t1dir 'rdyn_1.nii'], [t1dir 'rdyn_1_regcopy.nii'])

        t1 = double(load_nii([t1dir '3D_T1w.nii']).img);
        t1_nii = make_nii(t1, [1 1 1]);
        save_nii(t1_nii, [t1dir '3D_T1w_regcopy.nii']);
        centre_header_file([t1dir '3D_T1w_regcopy.nii']);

        copyfile([dcedir 'Patlak_' cut '/v_p.nii'],[msdir 'v_p_Patlak.nii'])
        copyfile([dcedir 'Patlak_' cut '/Ktrans.nii'],[msdir 'Ktrans_Patlak.nii'])

        % registration
        otherfn = {[msdir 'v_p_Patlak.nii'],
                   [msdir 'Ktrans_Patlak.nii']}
        register_x([t1dir '3D_T1w_regcopy.nii'],[t1dir 'rdyn_1_regcopy.nii'],otherfn)
    end

%% Cut anything outside brain from all maps
    for person = 1:length(id_list)
        id = char(id_list(person))
        t1dir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/structural/'];
        dcedir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/dce_fitting/'];
        dyndir = ['/Volumes/G-DRIVE/Stroke_Impact/' sitename '/' timepoint '/' id '/dynamic_series/'];
        % All registered Ktrans, vp maps
        t1dir,csfin,keeproi,roifn,mapcut,mapfn,r
        wholebraincut(t1dir,0,0,'',1,[msdir 'rKtrans_Patlak.nii'],0)
        wholebraincut(t1dir,0,0,'',1,[msdir 'rv_p_Patlak.nii'],0)
    end