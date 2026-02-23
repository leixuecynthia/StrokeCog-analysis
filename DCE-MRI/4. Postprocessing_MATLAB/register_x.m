function [  ] = register_x(reffn, movingfn,otherfn)
%-----------------------------------------------------------------------
% General function to register one image to another in SPM12
% spm SPM - SPM12 (7771)
%-----------------------------------------------------------------------
%%
clear jobs;
jobs{1}.spm.spatial.coreg.estwrite.ref = {[reffn ',1']};
jobs{1}.spm.spatial.coreg.estwrite.source = {[movingfn ',1']};
if length(otherfn)>0
    for i=1:length(otherfn)
        otherfn(i) = {[char(otherfn(i)) ',1']};
    end
    transpose(otherfn);
    jobs{1}.spm.spatial.coreg.estwrite.other = otherfn;
end
jobs{1}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';
jobs{1}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];
jobs{1}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
jobs{1}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];
jobs{1}.spm.spatial.coreg.estwrite.roptions.interp = 4;
jobs{1}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];
jobs{1}.spm.spatial.coreg.estwrite.roptions.mask = 0;
jobs{1}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';

spm('Defaults','pet');
spm_jobman('initcfg');
spm_jobman('run',jobs);
spm quit;
end