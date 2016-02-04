SGE Parallel library - Ver. 1
-----------------------------

Author: Nimrod Dorfman
Date: 26/6/2011


This library allows parallel execution on the Sun Grid Engine directly
from Matlab, provided that Matlab is running on a node of the cluster.


The library consists of the following program files:

- run_matlab_script
   Unix script that runs a Matlab command.
   The Matlab command can use the variable job_id, which stores the id of
   the job within the parallel job array.
   This script is what SGE eventually executes in parallel.

- sge_parallel.m
   Low level Matlab function for running parallel code.
   It can run parallel jobs without passing any runtime context to them.
   If the parallel code does not require any job-specific parameters except
   for the job id - this is the function to use.

- run_parallel.m
   High level Matlab function that can pass different variable values to
   different jobs. Number of jobs to run is set automatically according to
   number of variable values.

- delete_sge_logs.m
   Running parallel jobs with these functions will cause SGE to create
   output and error files in your home directory.
   These files are named ~/sge_parallel/sge_parallel.[e|o][task_id].[job_id].
   They are not deleted automatically, as they may be useful for debugging.
   This function will delete run_matlab_script.* from your home directory.


All functions are documented. Use help function_name (in Matlab) or just
look at the code for more details.

For an example see sge_demo.m.

