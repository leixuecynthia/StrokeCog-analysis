function [  ] = wmh_lpa(FLAIR_fn, T1_fn)
%-----------------------------------------------------------------------
% General function to segment white matter hyperintensities from FLAIR and T1w image using SPM12 - requires installation of the LST toolbox.  
% spm SPM - SPM12 (7771)
%-----------------------------------------------------------------------
    clear jobs;
    jobs{1}.spm.tools.LST.lpa.data_F2 = {[FLAIR_fn ',1']};
    jobs{1}.spm.tools.LST.lpa.data_coreg = {[T1_fn ',1']};
    jobs{1}.spm.tools.LST.lpa.html_report = 0;
    
    spm('Defaults','pet');
    spm_jobman('initcfg');
    spm_jobman('run',jobs);
    % oplist = spm_jobman('run',jobs);
    spm quit;
end