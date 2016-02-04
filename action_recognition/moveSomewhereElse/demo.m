startup;
baseDir = '~/storage/data/TUHOI';
imageNames = dir(fullfile(baseDir,'*.jpg'));
load('~/storage/misc/fra_classifiers.mat');
resultsPath = '~/storage/TUHOI_faces_and_features';
d = dir(fullfile(resultsPath,'*.mat'));

%% load results
clear allFeats;
for t = 1:length(d)
    t
    L = load(fullfile(resultsPath,d(t).name));
    allFeats(t) = L;
end

all_paths = {allFeats.imagePath};
all_deep_feats = {};
for t = 1:length(allFeats)
    all_deep_feats{t} = allFeats(t).deep_nn_face_feats(:,1);
end
all_deep_feats = cat(2,all_deep_feats{:});

face_scores = zeros(size(all_deep_feats,2),1);
for t = 1:length(allFeats)
    face_scores(t) = allFeats(t).faces.boxes(1,end);
end

face_scores = zeros(size(all_deep_feats,2),1);
for t = 1:length(allFeats)
    face_scores(t) = allFeats(t).faces.boxes(1,end);
end


% save allFeats allFeats

myNormalizeFun = @(x) normalize_vec(x);
%
feats_face = myNormalizeFun(all_deep_feats); % face features (selection)

all_face_images = {};
for t = 1:length(allFeats)
    all_face_images{t} = allFeats(t).face_images{1};
end
    
sel_ = find(face_scores > 1);
n = floor(length(sel_).^.5).^2;
U = mImage(all_face_images(sel_(1:n)));
    
    %% classify using learned classifiers...
    classifiers = [fra_classifiers.classifier];
    ws = classifiers(1:end-1,:);
    scores = ws'*feats_face;
    scores(:,face_scores < 1) = -inf;
    iClass = 4;
    addpath('/home/amirro/code/mircs/utils/');
    [u,iu] = sort(scores(iClass,:),'descend');
    %
    % [u,iu] = sort(face_scores,'descend');
    
    
    for t = 1:length(u)
        k = iu(t);
        I = getImage(conf,all_paths{k});
        clf; imagesc2(I); plotBoxes(allFeats(k).faces.boxes(1,:));
        pause
    end
    %
    % conf.baseDir = baseDir;
    % displayImageSeries(conf,all_paths(iu));
    
