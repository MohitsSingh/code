if (~exist('initialized','var'))
    initpath;
    config;        
    conf.get_full_image = true;
%    load ~/storage/misc/imageData_new;            
    newImageData = augmentImageData(conf,[]);
    addpath(genpath('/home/amirro/code/3rdparty/MatlabFns/'));    
    [learnParams,conf] = getDefaultLearningParams(conf,1024);    
    initialized = true;
end
