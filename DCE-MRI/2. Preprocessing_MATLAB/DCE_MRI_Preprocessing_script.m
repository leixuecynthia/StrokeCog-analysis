%% Script used for preprocessing DCE-MRI data from the StrokeCog-BBB study.
% Averages VFA repeat images, calculates B1 map, and fits T1 map using Madym.
% Seperates and aligns dynamic series using SPM12's 'realign'.

%% Add required paths
addpath('/Users/user/Documents/MATLAB/fm_toolbox')
addpath('/Users/user/Documents/MATLAB/spm12')
addpath('/Users/user/Documents/MATLAB/IMPaCT_fitting/Manchester') % Code directory
timepoint_folder = '6months'

id_file = fileread(['/Volumes/G-DRIVE/Stroke_Impact/Manchester/6month_IDs.txt']); % Participant ID file, text file in data directory
id_list = strsplit(id_file);
for person=1:length(id_list) % Loop over participants
    id = char(id_list(person))
    rootdir = fullfile('/Volumes/G-DRIVE/Stroke_Impact/Manchester',timepoint_folder,id)
    rawdir = fullfile(rootdir,'raw')
    cd(rawdir)
    folderlist = ls(rawdir); % dce image FA 1
    folderlist = string(folderlist);

    %% copy dynamic image series
    dyn_fn = 'WIP_DCE_SPIRAL_INJ8_12DEG'
    if isfile(fullfile(rawdir,[dyn_fn '.nii']))==1
        dyndir = fullfile(rootdir, 'dynamic_series')
        mkdir(dyndir)
        if ismember([dyn_fn 'a.nii'], folderlist)==1
            dyn_fn = [dyn_fn 'a.nii']
        else
            dyn_fn = [dyn_fn '.nii']
        end
        copyfile(dyn_fn, dyndir)
        cd(dyndir)
        movefile(dyn_fn, 'dce_inj.nii')
        cd(rawdir)

    %% Average over repeat Variable flip angle images
        vsize = [1.5 1.5 2]
        vfadir = fullfile(rootdir, 'vfa')
        mkdir(vfadir)
        for fa = [2,6,10,15]
            copyfile(['WIP_DCE_SPIRAL_' num2str(fa) 'DEG.nii'], vfadir)
            cd(vfadir)
            movefile(['WIP_DCE_SPIRAL_' num2str(fa) 'DEG.nii'], [num2str(fa) 'deg.nii'])
            deg_nii = load_nii(fullfile(vfadir, [num2str(fa) 'deg.nii']));
            deg = double(deg_nii.img);
            deg_cut = deg(:,:,:,2:6);
            deg_avg = mean(deg_cut, 4);
            deg_avg_nii = make_nii(deg_avg, vsize)
            save_nii(deg_avg_nii, fullfile(vfadir, [num2str(fa) 'deg.nii']))
            cd(rawdir)
        end
        deg12_nii = load_nii(fullfile(dyndir, 'dce_inj.nii'));
        deg12 = double(deg12_nii.img);
        deg12_cut = deg12(:,:,:,2:6);
        deg12_avg = mean(deg12_cut, 4);
        deg12_avg_nii = make_nii(deg12_avg, vsize)
        save_nii(deg12_avg_nii, fullfile(vfadir, ['12deg.nii']))
        
        %% Identify / calculate B1 map
        cd(rawdir)
        if isfile(fullfile(rawdir,'WIP_B1_map_SENSE_NEW_r25_e1.nii'))==1
            imgsTR2_nii = load_nii('WIP_B1_map_SENSE_NEW_r25_e1.nii')
            imgsTR2 = double(imgsTR2_nii.img);
            imgsTR1_nii = load_nii('WIP_B1_map_SENSE_NEW_r125_e1.nii')
            imgsTR1 = double(imgsTR1_nii.img);
        elseif isfile(fullfile(rawdir,'WIP_B1_map_SENSE_NEW_r25.nii'))==1
            imgsTR2_nii = load_nii('WIP_B1_map_SENSE_NEW_r25.nii')
            imgsTR2 = double(imgsTR2_nii.img);
            imgsTR1_nii = load_nii('WIP_B1_map_SENSE_NEW_r125.nii')
            imgsTR1 = double(imgsTR1_nii.img);
        elseif isfile(fullfile(rawdir, "WIP_B1_map_SENSE_NEW_e2.nii"))==0
            e1_nii = load_nii('WIP_B1_map_SENSE_NEW_e1.nii')
            e1_img = double(e1_nii.img);
            e1_dims = size(e1_img);
            e1a_nii = load_nii('WIP_B1_map_SENSE_NEW_e1a.nii')
            e1a_img = double(e1a_nii.img);
            e1a_dims = size(e1a_img);
            if length(e1_dims)==4;
                A = e1_img(:,:,:,1);
                B = e1_img(:,:,:,2);
                if B(48,63,35)>A(48,63,35)
                    imgsTR1 = A;
                    imgsTR2 = B;
                else
                    imgsTR1 = B;
                    imgsTR2 = A;
                end
            elseif length(e1a_dims)==4
                A = e1a_img(:,:,:,1);
                B = e1a_img(:,:,:,2);
                if B(48,63,35)>A(48,63,35)
                    imgsTR1 = A;
                    imgsTR2 = B;
                else
                    imgsTR1 = B;
                    imgsTR2 = A;
                end
            else
                e1b_nii = load_nii('WIP_B1_map_SENSE_NEW_e1b.nii');
                e1b_img = double(e1b_nii.img);
                imshow(e1b_img(:,:,35),[0 150])
                x = input('\n B1 map? (Yes = 1, No = 0) \n')
                if x==1
                    A = e1a_img;
                    B = e1b_img;
                else
                    imshow(e1a_img(:,:,35), [0,10])
                    y = input('\n B1 map? (Yes = 1, No = 0) \n')
                    if y==1
                        A = e1_img;
                        B = e1b_img;
                    else
                        A = e1_img;
                        B = e1a_img;
                    end
                end
                if B(48,63,35)>A(48,63,35)
                    imgsTR1 = A;
                    imgsTR2 = B;
                else
                    imgsTR1 = B;
                    imgsTR2 = A;
                end
            end
        elseif isfile(fullfile(rawdir,'WIP_B1_map_SENSE_NEW_e1a.nii'))==0
            e1_nii = load_nii('WIP_B1_map_SENSE_NEW_e1.nii')
            e1_img = double(e1_nii.img);
            e1_dims = size(e1_img);
            e1a_nii = load_nii('WIP_B1_map_SENSE_NEW_e2.nii')
            e1a_img = double(e1a_nii.img);
            e1a_dims = size(e1a_img);
            if length(e1_dims)==4;
                A = e1_img(:,:,:,1);
                B = e1_img(:,:,:,2);
                if B(48,63,35)>A(48,63,35)
                    imgsTR1 = A;
                    imgsTR2 = B;
                else
                    imgsTR1 = B;
                    imgsTR2 = A;
                end
            elseif length(e1a_dims)==4
                A = e1a_img(:,:,:,1);
                B = e1a_img(:,:,:,2);
                if B(48,63,35)>A(48,63,35)
                    imgsTR1 = A;
                    imgsTR2 = B;
                else
                    imgsTR1 = B;
                    imgsTR2 = A;
                end
            else
                e1b_nii = load_nii('WIP_B1_map_SENSE_NEW_e1b.nii');
                e1b_img = double(e1b_nii.img);
                imshow(e1b_img(:,:,35),[0 150])
                x = input('\n B1 map? (Yes = 1, No = 0) \n')
                if x==1
                    A = e1a_img;
                    B = e1b_img;
                else
                    imshow(e1a_img(:,:,35), [0,10])
                    y = input('\n B1 map? (Yes = 1, No = 0) \n')
                    if y==1
                        A = e1_img;
                        B = e1b_img;
                    else
                        A = e1_img;
                        B = e1a_img;
                    end
                end
                if B(48,63,35)>A(48,63,35)
                    imgsTR1 = A;
                    imgsTR2 = B;
                else
                    imgsTR1 = B;
                    imgsTR2 = A;
                end
            end
        else
            A_nii = load_nii('WIP_B1_map_SENSE_NEW_e1a.nii')
            A = double(A_nii.img);
            B_nii = load_nii('WIP_B1_map_SENSE_NEW_e1.nii')
            B = double(B_nii.img);
            if B(48,63,35)>A(48,63,35)
                imgsTR1 = A;
                imgsTR2 = B;
            else
                imgsTR1 = B;
                imgsTR2 = A;
            end
        end

        % Calculate B1 map
        
        TR1 = 25.0; %ms
        TR2 = 125.0; %ms
        sImgsTR1 = smooth3(imgsTR1,'box',[5 5 3]);
        sImgsTR2 = smooth3(imgsTR2,'box',[5 5 3]);
        r = imgsTR1./imgsTR2;
        sr = sImgsTR1./sImgsTR2;
        n = TR2/TR1;
         
        FA_map = acosd((r.*n - 1)./(n - r))/60; % B1 map
        sFA_map = acosd((sr.*n - 1)./(n - sr))/60; %smoothed B1 map
        sFA_map = sFA_map*100;
        
        nii_TR1 = make_nii(sImgsTR1, vsize)
        save_nii(nii_TR1, fullfile(vfadir, ['sImgsTR1.nii']))
        nii_TR2 = make_nii(sImgsTR2, vsize)
        save_nii(nii_TR2, fullfile(vfadir, ['sImgsTR2.nii']))
        nii_B1 = make_nii(sFA_map, vsize)
        save_nii(nii_B1, fullfile(vfadir, ['sB1.nii']))
        cd(rawdir)
    
        %% Fit T1 map using Madym
       base_pn =  '/Volumes/G-DRIVE/Stroke_Impact/Manchester/';
       pn_base = fullfile(base_pn, timepoint_folder, id)
       VFAs = [2, 6, 10, 12, 15]; % FAs for VFA
    
          for i = 1:length(VFAs)
            filename = [num2str(VFAs(i)) 'deg.nii'];
            filepaths(i) = {fullfile(pn_base, 'vfa', filename)};
          end
        run_madym_MakeXtr(...
        'T1_vols', [filepaths], ...
        'working_directory', pn_base, ...
        'dynamic_basename', 'dynamic_series/rdyn_', ...
        'sequence_format', '%01u', ...Format for converting dynamic series index to string, eg %01u
        'sequence_start', 1, ...Start index for dynamic series file names
        'sequence_step', 1, ...Step between indexes of filenames in dynamic series
        'n_dyns', 157, ...
        'make_t1', 1, ...
        'make_dyn', 1, ...  
        'temp_res', 7.64, ... %either set this or specify dyn_times
        'TR', 10.6, ... 
        'FA', 12, ... 
        'VFAs', VFAs, ... 
        'dummy_run', 0);
    
        B1_input = fullfile(pn_base, 'vfa', 'sB1.nii');
        VFAs = [2, 6, 10, 12, 15]; % FAs for VFA
        audit_dir = fullfile(pn_base, 'madym_audit_logs')
        
        % Create filenames for VFA
        for i = 1:length(VFAs)
            filename = [num2str(VFAs(i)) 'deg.nii'];
            filepaths(i) = {fullfile(pn_base, 'vfa', filename)};
        end
        
        % Run Madym T1
        run_madym_T1(...
            'cmd_exe', [local_madym_root 'madym_T1'],...
            'T1_vols', [filepaths],... Cell array of variable flip angle file paths
            'ScannerParams', [],... either single vector used for all samples, or 2D array, 1 row per sample
            'signals', [],...Signals associated with each FA, 1 row per sample
  	        'TR', 10.6,... TR in msecs, required if directly fitting (otherwise will be taken from FA map headers);
            'method', 'VFA_B1',...T1 method to use to fit, see notes for options
            'B1_name', B1_input,...Path to B1 correction map
            'B1_correction', false, ... Apply B1 correction
            'B1_scaling', 100, ... Scaling factor to use with B1 map
            'output_dir', [fullfile(pn_base,'t1_map')], ...Output path, will use temp dir if empty;
            'noise_thresh', 100000, ... PD noise threshold
            'roi_name', [],...Path to ROI map
            'error_name', [],... Name of error codes image
            'img_fmt_r', '',...Set image read format
            'img_fmt_w', '',...Set image write format
            'no_audit', NaN,... Turn off audit log
            'no_log', NaN,... Turn off propgram log
            'quiet', NaN,... Suppress output to stdout
            'program_log_name', '',...Program log file name
            'config_out', '',...Name of output config file
            'audit_dir', audit_dir,...Folder in which audit logs are saved
            'overwrite', true,...Set overwrite existing analysis in output dir ON
            'working_directory', '',...Sets the current working directory for the system call, allows setting relative input paths for data
            'dummy_run', false ...Don't run any thing, just print the cmd we'll run to inspect
            );
    
        %% Seperate & Align dynamic series using SPM12
        % split dynamic series for alignment
        fdce = 'dce_inj.nii'
        pn_dce = fullfile(base_pn, timepoint_folder, id, 'dynamic_series');
        fn_dce = (fullfile(pn_dce, fdce));
        cd(pn_dce);
        dce_im = load_nii(fn_dce);
        dce = double(dce_im.img);
        dims = size(dce);
    
        for d = 1:dims(4)
            dce_dyn = dce(:,:,:,d);
            dyn_nii = make_nii(dce_dyn, vsize);
            dyn_nii_name = ['dyn_' num2str(d) '.nii'];
            save_nii(dyn_nii, dyn_nii_name);
            centre_header_file(dyn_nii_name);
            dce_filepaths(d) = {fullfile(pn_dce,dyn_nii_name)};
        end
        dce_filepaths = transpose(dce_filepaths)
        % Align
        SPM_align_MAN(dce_filepaths);
        clear jobs;
        clear dce_filepaths
    else
        cd(rootdir)
        fileID = fopen('NoDynamicSeries.txt','w');
        fprintf(fileID,'No DCE dynamic series found');
        fclose(fileID);
    end
end