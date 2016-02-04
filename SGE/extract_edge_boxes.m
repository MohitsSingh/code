function res = extract_edge_boxes(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd ~/code/mircs;
    initpath;
    config;
    vl_setup;
    res.conf = conf;
    
    model=load('/home/amirro/code/3rdparty/edgeBoxes/models/forest/modelBsds'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    
    opts = edgeBoxes;
    opts.alpha = .65;     % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 1e3;  % max number of boxes to detect
    res.model = model;
    res.opts = opts;
    
    return
end

model = initData.model;
conf = initData.conf;
opts = initData.opts;

curPath = params.path;
I = imread2(curPath);
curBoxes = edgeBoxes(I,model,opts);
curBoxes = curBoxes(:,1:4);
curBoxes(:,3:4) = single(curBoxes(:,3:4)+curBoxes(:,1:2));

nBoxesPerImage=50;
% sample a subset of the boxes as images and keep them
resizer = @(y) cellfun2(@(x) imResample(x,[32 32]),y);
curBoxes_sub = curBoxes(vl_colsubset(1:size(curBoxes,1),nBoxesPerImage,'Uniform'),:);
imgs = resizer(multiCrop2(I,curBoxes_sub));




res = struct('imagePath',curPath,'boxes',curBoxes,'imgs',cat(4,imgs{:}));

