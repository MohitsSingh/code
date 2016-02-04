function res = upper_body_2_parallel(conf,I,reqInfo,moreParams)

if (nargin == 0)
    cd ~/code/mircs/;
    initpath;
    config;
    res.conf = conf;        
    cd '/home/amirro/code/3rdparty/voc-release5';
    startup;
    ubcModelFile = './models/mmModel_dsc.mat';
    dpmModelFile = './models/ubDet_permuteDsc_nComp-2_nPart-2_cascade.mat';
    ubc = load(ubcModelFile, 'mmModel'); % load UBC model
    % 4-UB configurations yield no benefit, remove them to improve detection speed
    ubc.mmModel.cmModels(4) = [];
    % load DPM model and cascade version to compute dense scores
    dpm = load(dpmModelFile, 'cscModel', 'model');
    res.dpm = dpm;res.ubc =ubc;    
    
    return;
end
conf_ = reqInfo.conf;

I = getImage(conf_,I);if (length(size(I))==2), I = cat(3,I,I,I);end
resizeFactor = 2;
I = imresize(I,resizeFactor,'bilinear');
threshold = -1;
reqInfo.dpm.model.thresh = -2;
reqInfo.dpm.cscModel.thresh = -2;
% reqInfo.ubc.mmModel.cmModels
res = MUB_UbDet.ubcCascadeDetect(I, reqInfo.dpm.model, reqInfo.dpm.cscModel, reqInfo.ubc.mmModel);
if (any(res))
    res = res(:, res(5,:) >= threshold)';
    res(:,1:4) = res(:,1:4)/resizeFactor;
    res = struct('boxes',res);
else
    res = struct('boxes',[]);
end
