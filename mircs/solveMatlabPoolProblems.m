% distcomp.feature(LocalUseMpiexec,'false');
% 
% 
% pp = prefdir;%/home/amirro/.matlab/R2010b

sched = findResource('scheduler', 'type', 'local')
set(sched,'DataLocation','/home/amirro/matlab_sched/');