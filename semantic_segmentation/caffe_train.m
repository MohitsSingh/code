function caffe_train(caffe_root,caffe_solver_file,gpuID,log_file,caffe_model_to_finetune,snapName)


cmd=[caffe_root, '/build/tools/caffe train ', sprintf('-solver %s', caffe_solver_file)];
if nargin > 5
    cmd = [cmd, ' -snapshot ',snapName];
end
if gpuID>-1
    cmd=sprintf('%s -gpu %d',cmd,gpuID);
end
if nargin < 5
    if exist('caffe_model_to_finetune','var')
        cmd=sprintf('%s -weights %s',cmd,caffe_model_to_finetune);
    end
end
if exist('log_file','var') && ~isempty(log_file)
    if 1
        % csh
        cmd=sprintf('%s |& tee %s',cmd,log_file);
    elseif 0
        % csh
        cmd=sprintf('%s >& %s',cmd,log_file);
    else
        % bash
        cmd=sprintf('%s 2> %s',cmd,log_file);
    end
end
system(cmd);
