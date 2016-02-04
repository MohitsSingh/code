

% Run parallel code on the Sun Grid
%
% Note:
%   Can be used from any linux workstation.
%   Requires ssh connection to cluster without password.
%
% Syntax:
%   sge_parallel(matlab_code, num_jobs)
%   sge_parallel(matlab_code, num_jobs, first_job_id)
%   sge_parallel(..., cluster_name)
%   sge_parallel(..., 'async')
%   sge_parallel(..., 'debug')
%
% Input:
%   matlab_code -
%       The code to run, as a string.
%       Code can access the variable 'job_id' to get the process id.
%   num_jobs -
%       Number of jobs to run.
%   first_job_id (optional) -
%       Id of first job.
%       Other jobs will have consequitive id numbers.
%       Default is 1.
%   cluster_name (optional) -
%       Name of cluster to use.
%       Cluster is accessed via ssh without specifying password.
%       Default is 'mcluster01'.
%   'async' (optional) -
%       If specified, send the job and return immediately (asynchronously).
%       Otherwise wait for all jobs to finish before returning.
%   'debug' (optional) -
%       If specified qsub output will be displayed.
%
% Output:
%   qsub log files are saved as
%   ~/sge_parallel/run_matlab_script.[e|o][task_id].[job_id]
%
function sge_parallel(matlab_code, num_jobs, varargin)

% parse params
[cluster_name, first_job_id, sync, debug] = parse_params(varargin);

% constants
qsub_out_dir = '~/sge_parallel_new/';

% run parallel Matlabs
if sync
    sync_str = 'y';
else
    sync_str = 'n';
end
last_job_id = first_job_id + num_jobs - 1;
script_file = fullfile(fileparts(mfilename('fullpath')), 'run_matlab_script');
%-q 32g.q (for cluster 3)
%-q amd64g.q (for cluster 1)
cmd = sprintf('ssh -A %s qsub -sync %s -t %0.0f-%0.0f -o %s -e %s %s %s', ...
    cluster_name, sync_str, first_job_id, last_job_id, qsub_out_dir, qsub_out_dir, script_file, quote_text(quote_text(matlab_code)));
if debug
    disp(['Executing: ' cmd]);
    [status, result] = system(cmd, '-echo');
else
    [status, result] = system(cmd);
end
if status
    warning('Error in qsub:\n%s', result);
end



% parse input
function [cluster_name, first_job_id, sync, debug] = parse_params(params)

% defaults
first_job_id = 1;
sync = true;
debug = false;
cluster_name = 'mcluster03';

for i = 1:numel(params)
    if ischar(params{i})
        if strcmpi(params{i}, 'async')
            sync = false;
        elseif strcmpi(params{i}, 'debug')
            debug = true;
        else
            cluster_name = params{i};
        end
    else
        assert((i == 1) && isnumeric(params{i}) && (numel(params{i}) == 1), ...
            'Invalid parameter value');
        first_job_id = params{i};
    end
end


% quote text for linux:
% surround with single quotes
% replace internal ' with '"'"'
function text = quote_text(text)
text = ['''' strrep(text, '''', '''"''"''') ''''];

