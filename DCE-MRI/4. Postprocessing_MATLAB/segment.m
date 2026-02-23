function [  ] = segment(fn)
%-----------------------------------------------------------------------
% General function to segment T1w image into GM, WM, CSF using SPM12
% spm SPM - SPM12 (7771)
%-----------------------------------------------------------------------
clear jobs;
jobs{1}.spm.spatial.preproc.channel.vols = {char(fn)};
jobs{1}.spm.spatial.preproc.channel.biasreg = 0.001;
jobs{1}.spm.spatial.preproc.channel.biasfwhm = 60;
jobs{1}.spm.spatial.preproc.channel.write = [0 0];
jobs{1}.spm.spatial.preproc.tissue(1).tpm = {'/Users/user/Documents/MATLAB/spm12/tpm/TPM.nii,1'};
jobs{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
jobs{1}.spm.spatial.preproc.tissue(1).native = [1 0];
jobs{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
jobs{1}.spm.spatial.preproc.tissue(2).tpm = {'/Users/user/Documents/MATLAB/spm12/tpm/TPM.nii,2'};
jobs{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
jobs{1}.spm.spatial.preproc.tissue(2).native = [1 0];
jobs{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
jobs{1}.spm.spatial.preproc.tissue(3).tpm = {'/Users/user/Documents/MATLAB/spm12/tpm/TPM.nii,3'};
jobs{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
jobs{1}.spm.spatial.preproc.tissue(3).native = [1 0];
jobs{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
jobs{1}.spm.spatial.preproc.tissue(4).tpm = {'/Users/user/Documents/MATLAB/spm12/tpm/TPM.nii,4'};
jobs{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
jobs{1}.spm.spatial.preproc.tissue(4).native = [1 0];
jobs{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
jobs{1}.spm.spatial.preproc.tissue(5).tpm = {'/Users/user/Documents/MATLAB/spm12/tpm/TPM.nii,5'};
jobs{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
jobs{1}.spm.spatial.preproc.tissue(5).native = [1 0];
jobs{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
jobs{1}.spm.spatial.preproc.tissue(6).tpm = {'/Users/user/Documents/MATLAB/spm12/tpm/TPM.nii,6'};
jobs{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
jobs{1}.spm.spatial.preproc.tissue(6).native = [0 0];
jobs{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
jobs{1}.spm.spatial.preproc.warp.mrf = 1;
jobs{1}.spm.spatial.preproc.warp.cleanup = 1;
jobs{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
jobs{1}.spm.spatial.preproc.warp.affreg = 'mni';
jobs{1}.spm.spatial.preproc.warp.fwhm = 0;
jobs{1}.spm.spatial.preproc.warp.samp = 3;
jobs{1}.spm.spatial.preproc.warp.write = [0 0];
jobs{1}.spm.spatial.preproc.warp.vox = NaN;
jobs{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
spm('Defaults','pet');
spm_jobman('initcfg');
spm_jobman('run',jobs);
% oplist = spm_jobman('run',jobs);
spm quit;
end