function res = run_drfi_saliency(conf,I,reqInfo,toIgnore)
if (nargin == 0)    
    cd /home/amirro/code/3rdparty/drfi_code_0.2
    addpath(genpath('.'));
    
    res.para = makeDefaultParameters;
    return;
end
I = imread(I);
res.smap = drfiGetSaliencyMap( I, reqInfo.para );
end