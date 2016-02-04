%% %%%%% Experiment 0028 %%%%%
% 24/2/2014 : train a deformable part model to find straws, cup, bottles,
% in the face area, where the given input images are only those with high
% scoring faces.

% initialization of paths, configurations, etc

if (~exist('initialized','var'))
    initpath;
    config;
    initialized = true;
    conf.get_full_image = true;
    load ~/storage/misc/imageData_new;
    dpmPath = '/home/amirro/code/3rdparty/voc-release4.01';
    addpath(genpath(dpmPath));
    newImageData = augmentImageData(conf,newImageData);
end

% run the visual phrases on everything.
curDir = pwd;
cd ~/code/SGE
extraInfo.conf = conf;
load ~/data/dpm_models/person_drinking_bottle_final.mat;
models{1} = model;
load ~/data/dpm_models/bottle_final.mat;
models{2} = model;
extraInfo.models = models;
extraInfo.path = path;
%extraInfo.dpmVersion = 4;
extraInfo.newImageData = newImageData;
% delete ~/sge_parallel_new/*;
job_suffix = 'phrases';
justTesting = false;
outDir = '~/storage/s40_drink_phrase';
detections = run_and_collect_results({newImageData.imageID},'detect_dpm_parallel',justTesting,extraInfo,job_suffix,[],outDir);
detPath = fullfile(outDir,'all.mat');
save(detPath,'detections');
%%
phrase_boxes = -inf*ones(length(detections),5);
bottle_boxes = -inf*ones(length(detections),5);

upper_body_boxes = -inf*ones(length(detections),5);

for k = 1:length(detections)
    %     I = getImage(conf,newImageData(k).imageID);
    b = detections{k}(1);
    if (~isempty(b.boxes))
        phrase_boxes(k,:) = b.boxes(1,:);
    end
    b = detections{k}(2);
    if (~isempty(b.boxes))
        bottle_boxes(k,:) = b.boxes(1,:);
    end
    if (~isempty(newImageData(k).upperBodyDets))
        upper_body_boxes(k,:) = newImageData(k).upperBodyDets(1,:);
    end
end
ids = {newImageData.imageID};

phrase_scores = phrase_boxes(:,5);
bottle_scores = bottle_boxes(:,5);

upper_body_scores  = upper_body_boxes(:,5);


ints = BoxIntersection(upper_body_boxes,phrase_boxes);

%%

theBoxes = bottle_boxes;
% intersects = ints(:,1)~=-1;
% theScores = bottle_scores*.1+phrase_scores+.01*upper_body_scores;
% theScores(~intersects) = -inf;
theBoxes = upper_body_boxes;
theScores = upper_body_scores;

close all;
[r,ir] = sort(theScores,'descend');


subs = {};
for ik = 1:length(r)
    k = ir(ik);
    %if k < 801 || k > 900, continue; end
    if (isTrain(k) && labels(k))
        I = getImage(conf,ids{k});
        subs{end+1} = cropper(I,round(theBoxes(k,:)));continue;        
    end
    continue;
    clf; imagesc(I); axis image; hold on; plotBoxes(theBoxes(k,:),'g','LineWidth',2);
    pause
end

A = cellfun2(@(x) imResample(x,[80 80],'bilinear'),subs);
AA = cellfun2(@rgb2gray,A);
% figure,imshow(sum(cat(3,AA{:}),3),[]);
%displayImageSeries(conf,ids(ir));

%%
% allRes = {};
is_valid = true(size(newImageData));
labels = [newImageData.label];
faceScores = [newImageData.faceScore];
faceBoxes = zeros(length(newImageData),4);
lm = [newImageData.faceLandmarks];
is_valid = faceScores >=-.5;
isTrain = [newImageData.isTrain]; nTrain = nnz(isTrain); nTest = nnz(~isTrain);
train_sel = isTrain & is_valid;
test_sel = ~isTrain & is_valid;
finalScores =-inf*ones(length(newImageData),1);
% finalScores(test_sel) = sum(dpm_scores(test_sel,:),2);
finalScores(test_sel) = max(theScores(test_sel,:),[],2);
% finalScores(test_sel) = f;%sum(dpm_scores(test_sel,[1 2 3]),2);
% showSorted(subImages(test_sel),finalScores(test_sel),50);
sel_ = ~[newImageData.isTrain];
labels_ = [newImageData.label]; %labels_ = labels_(sel_);
% finalScores = finalScores(sel_);
vl_pr(2*labels_(test_sel)-1,finalScores(test_sel));
% vl_pr(2*labels_-1,finalScores);


% detections = cat(1,detections{:});
% imageIndices = cat(1,imageIndices{:});
% [s,is] = sort(imageIndices);
% detections = detections(is);
% cd(curDir);