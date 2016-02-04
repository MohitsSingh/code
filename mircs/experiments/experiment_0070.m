%%%%% Experiment 0070 - extract features from segmented people
if (~exist('initialized','var'))
    cd ~/code/mircs
    initpath;
    config;
    %addpath('/home/amirro/code/3rdparty/edgeBoxes/');
    addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
    % rmpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta12/'));
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');   
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;    
    addpath('/home/beny/code/3rd/deepLab/deepLabScripts/matlab/');
    s40_fra= s40_fra_faces_d;
    initialized = true;
end

isTrain = [s40_fra.isTrain];

featureExtractor = DeepFeatureExtractor(conf);

% get segmentation for each image.
cfrOutputDir = '/home/amirro/code/3rdparty/my_deeplab_scripts/workdir/res_W5_XStd50_RStd3_PosW3_PosXStd3/';
d = dir('/home/amirro/code/3rdparty/my_deeplab_scripts/workdir/res_W5_XStd50_RStd3_PosW3_PosXStd3/*.bin');
HUMAN = 15;
s40_fra(1).seg = [];
s40_fra(1).I = [];
s40_fra(1).I_rect = [];
%%
for t = 1:length(s40_fra)
    if mod(t,50)==0
        disp(t)
    end
    imgData = s40_fra(t);
    if (imgData.classID~=conf.class_enum.DRINKING),continue,end
    [I,I_rect] = getImage(conf,imgData);
    crfBinFile = j2m(cfrOutputDir,imgData,'.bin');
    map = LoadBinFile(crfBinFile, 'int16');
    curSeg = imResample(map==HUMAN,size2(I),'nearest');
    imgData.I = im2uint8(I);
    imgData.seg = curSeg;
    imgData.I_rect = I_rect;
    s40_fra(t) = imgData;
        clf; sc(cat(3,curSeg,I),'prob');
        dpc;
end
%%
x2({s40_fra(1:50:end).seg});
% feature set 1: appearance
s40_fra(1).seg_is_empty = false;


masks = {};

for t = 1:length(s40_fra)
    t
    imgData = s40_fra(t);
    curSeg = imgData.seg;    
    seg_cropped = cropper(curSeg,imgData.I_rect);
    s40_fra(t).seg_is_empty = none(seg_cropped);         
    masks{t} = seg_cropped;    
%     s40_fra(t).seg_is_empty
end


valids = ~[s40_fra.seg_is_empty];
mask_shape_feats = featureExtractor.extractFeaturesMulti(masks);

masked_appearance_feats = {};
for t = 1:length(s40_fra)
    t
    if(valids(t))
        masked_appearance_feats{t} = featureExtractor.extractFeaturesMulti_mask(s40_fra(t).I,{s40_fra(t).seg});
    end
end

croppedImages = multiCrop2({s40_fra.I},cat(1,s40_fra.I_rect));
cropped_appearance_feats = featureExtractor.extractFeaturesMulti(croppedImages);


save ~/storage/misc/human_action_seg.mat mask_shape_feats valids masked_appearance_feats cropped_appearance_feats

%curFeats = mask_shape_feats;
for u = 1:length(valids)
    if ~valids(u)
        masked_appearance_feats{u} = -inf(4096,1);
    end
end

curFeats = cat(2,masked_appearance_feats{:});

isTrain = valids & [s40_fra.isTrain];
isTest = ~[s40_fra.isTrain];
classes = [s40_fra.classID];
train_params.task = 'classification';
train_params.toBalance = -1;
train_params.lambdas=  [1e-2 1e-3 1e-5 1e-6 1e-7];% 1e-6]
learnedClassifiers = train_classifiers( curFeats(:,isTrain),classes(isTrain),[],train_params);
params.classes = conf.class_enum.DRINKING;
res = apply_classifiers(learnedClassifiers,curFeats(:,isTest),classes(isTest),params,false);

T = croppedImages(isTest);
% T = masks(isTest);
[r,ir] = sort(res.curScores,'descend');
displayImageSeries(conf,T(ir));
showSorted(croppedImages(isTest),res.curScores);

classes = 9;
nTotalClasses = length(classes);
avg_prec_est = zeros(nTotalClasses,length(train_features));
lambdas =  [1e-5 1e-6 1e-7];% 1e-6]
train_valids = valids(1:length(train_ids));
toBalance = 0;
for iClass = 1:nTotalClasses
    train_params.classes = classes(iClass);
    for iSubset = 19:length(train_features)
        feature_subset = iSubset
        train_features_1 = transform_features(train_features(feature_subset),train_params.features);
        %res_train = train_classifiers(train_features_1(1:1:end,train_valids),train_labels(train_valids),train_params,toBalance,lambdas);
        
        train_classifiers( train_data,train_labels,train_values,train_params)
        
        avg_prec_est(iClass,iSubset) = res_train.classifier_data.optAvgPrec;
        clf; figure(1); imagesc(avg_prec_est(:,19:end)); drawnow
    end
end
train_classifiers

size(a)
imagesc(a)

a(:,447)

for t = 1:length(s40_fra)
    
    imgData = s40_fra(t);
    if imgData.classID ~= conf.class_enum.SMOKING,continue,end
    [I,I_rect] = getImage(conf,imgData);    
    curSet = imgData.I;
    clf; sc(cat(3,imgData.seg,I),'prob');
        dpc;
end

