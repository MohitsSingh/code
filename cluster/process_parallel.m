function [ output_args ] = process_parallel( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
    useCurrentDir = false;

    cwd = '';
    if useCurrentDir
        cwd = '-cwd';

    cmd = sprintf('qsub %s',...
        cwd);

end

