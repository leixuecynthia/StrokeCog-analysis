function [  ] = SPM_align_MAN(dce_filepaths)
%-----------------------------------------------------------------------
% General function to align a dynamic series of images using SPM12.
% spm SPM - SPM12 (7771)
%-----------------------------------------------------------------------
%%
clear jobs;
jobs{1}.spm.spatial.realign.estwrite.data = {dce_filepaths}';
%%
jobs{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
jobs{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
jobs{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
jobs{1}.spm.spatial.realign.estwrite.eoptions.rtm = 1;
jobs{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
jobs{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
jobs{1}.spm.spatial.realign.estwrite.eoptions.weight = '';
jobs{1}.spm.spatial.realign.estwrite.roptions.which = [2 1];
jobs{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
jobs{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
jobs{1}.spm.spatial.realign.estwrite.roptions.mask = 1;
jobs{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';

spm('Defaults','pet');
spm_jobman('initcfg');
spm_jobman('run',jobs);
% oplist = spm_jobman('run',jobs);
spm quit;
end