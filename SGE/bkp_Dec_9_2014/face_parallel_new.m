function res = face_parallel_new(conf,I,reqInfo)

if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    load ~/storage/misc/upper_bodies.mat;
    res.data = data;
    res.conf = conf;
    return;
end
%% 2. Saliency Map Calculation
%%
data = reqInfo.data;
curData = data(findImageIndex(data,I));
if (~curData.isvalid)
    res = [];
else
    I = curData.subs{1};
    resizeFactor = 200/size(I,1);
    I = imresize(I,resizeFactor,'bilinear');
    [~,res] = face_detection(I);
        
    if (~isempty(res))
        res(:,1:4) = res(:,1:4)/resizeFactor;
    end
end