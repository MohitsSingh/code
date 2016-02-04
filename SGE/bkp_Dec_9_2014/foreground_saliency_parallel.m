function res = foreground_saliency_parallel(conf,I,res)

if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
    %% 1. Parameter Settings
    res.doFrameRemoving = true;
    res.useSP = true;
    res.conf = conf;
    return;
end
%% 2. Saliency Map Calculation
%%
res = struct('sal',{},'sal_bd',{},'bbox',{},'resizeRatio',{});
opts.show = false;
maxImageSize = 300;
opts.maxImageSize = maxImageSize;
spSize = 40;
opts.pixNumInSP = spSize;
conf.get_full_image = true;
[I,I_rect] = getImage(conf,I);
n = 0;
for t = [1 .8 .5 .3];
    t
    
    bbox = [1 1 size(I,2) size(I,1)];
    bbox = round(inflatebbox(bbox,[1 t],'post',false));
    I_ = cropper(I,round(bbox));
    [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I_),opts);
    n = n+1;
    res(n).sal = sal;
    res(n).sal_bd = sal_bd;
    res(n).resizeRatio = resizeRatio;
    res(n).bbox = bbox;
    % %     clf,subplot(1,2,1);imagesc2(I_);
    % %     subplot(1,2,2),imagesc2(sal);
    % %     drawnow;pause
end
