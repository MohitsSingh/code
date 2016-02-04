% Run parallel code on the Sun Grid
%
% Note: Should only be used when running on a cluster node.
%
% Syntax:
%   sge_parallel(matlab_code, num_jobs, first_job_id, 'async', 'debug')
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
%   'async' (optional) -
%       If specified, send the job and return immediately (asynchronously).
%       Otherwise wait for all jobs to finish before returning.
%   'debug' (optional) -
%       If specified qsub output will be displayed.
%
function sge_parallel(matlab_code, num_jobs, varargin)

% parse params
[first_job_id, sync, debug] = parse_params(varargin);

% constants
qsub_out_dir = '~/sge_parallel/';

% run parallel Matlabs
if sync
    sync_str = 'y';
else
    sync_str = 'n';
end
last_job_id = first_job_id + num_jobs - 1;
script_file = fullfile(fileparts(mfilename('fullpath')), 'run_matlab_script');
cmd = sprintf('qsub -sync %s -t %0.0f-%0.0f -o %s -e %s %s "%s"', ...
    sync_str, first_job_id, last_job_id, qsub_out_dir, qsub_out_dir, script_file, matlab_code);
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
function [first_job_id, sync, debug] = parse_params(params)

% defaults
first_job_id = 1;
sync = true;
debug = false;

for i = 1:numel(params)
    if ischar(params{i})
        if strcmpi(params{i}, 'async')
            sync = false;
        elseif strcmpi(params{i}, 'debug')
            debug = true;
        end
    else
        assert((i == 1) && isnumeric(params{i}) && (numel(params{i}) == 1), ...
            'Invalid parameter value');
        first_job_id = params{i};
    end
end

