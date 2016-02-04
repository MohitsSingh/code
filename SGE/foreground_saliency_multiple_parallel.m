function res = foreground_saliency_multiple_parallel(conf,I,reqInfo)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
    %% 1. Parameter Settings
    res.doFrameRemoving = true;
    res.useSP = true;
    res.conf = conf;
    load fra_db;
    res.fra_db = fra_db;
    return;
end
%% 2. Saliency Map Calculation
%%
opts.show = false;
maxImageSize = 150;
opts.maxImageSize = maxImageSize;
spSize = 15;
opts.pixNumInSP = spSize;
conf.get_full_image = true;

fra_db = reqInfo.fra_db;
all_class_names = {fra_db.class};
class_labels = [fra_db.classID];
classes = unique(class_labels);
% make sure class names corrsepond to labels....
[lia,lib] = ismember(classes,class_labels);
classNames = all_class_names(lib);
roiParams.useCenterSquare = false;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
%[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);

% [I,I_rect] = getImage(conf,I);
n = 0;
res = struct('sal',{},'sal_bd',{},'bbox',{},'resizeRatio',{});

for t = [.2 .3 .5 .7]
    t
    
    
    
    roiParams.infScale = t*3.5;
    roiParams.absScale = 200*roiParams.infScale/2.5;
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
    %     I = getImage(conf,curImageData);
    I = imresize(I,[300 NaN]);
    [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I),opts);
    n = n+1;
    res(n).sal = sal;
    %         clf;imagesc2(sal);
    %         disp('hit any key to continue');
    %         pause;
    res(n).sal_bd = sal_bd;
    %     res(n).resizeRatio = resizeRatio;
    %     res(n).bbox = roiBox;
end
