% Delete SGE log files
%
% Running parallel jobs with these functions will cause SGE to create
% output and error files in your home directory.
% These files are named run_matlab_script.[e|o][task_id].[job_id].
% They are not deleted automatically, as they may be useful for debugging.
% This function will delete run_matlab_script.* from your home directory.
%
% Syntax:
%   delete_sge_logs()
%
function delete_sge_logs()

% constants
qsub_out_files = '~/sge_parallel/sge_parallel.o*';   % output files
qsub_err_files = '~/sge_parallel/sge_parallel.e*';   % error files

delete(qsub_out_files);
delete(qsub_err_files);

