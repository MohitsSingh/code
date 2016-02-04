%%%%%% Experiment 7 %%%%%%%
% Oct. 29, 2013

% just like experiment 2, but with more training data, more clusters, etc. 
%ah, i've been using the wrong feature vector. switching to felzenszwalb
%and doing it again.

function experiment_0007
initpath;
config;
resultDir = ['~/mircs/experiments/' mfilename];
ensuredir(resultDir);
conf.features.vlfeat.cellsize = 8;
baseDir = '~/storage/train_faces/aflw_cropped_context/';
d = dir(fullfile(baseDir,'*.jpg'));

pos_train_images = d(1:2:end);
pos_val_images = d(2:2:end);

ims = {};
for k = 1:length(pos_train_images)
    if (mod(k,100)==0)
        k
    end
    I = imread(fullfile(baseDir, pos_train_images(k).name));
    I = I(round(.25*end:.75*end),round(.25*end:.75*end),:);
     II = imResample(I,[80 80],'bilinear');        
%     I = I(64:127,64:127,:);
%     clf; imshow(I); drawnow;
%     size(I)
    
    if (length(size(II))==2)
        II = repmat(II,[1 1 3]);
    end
    ims{k} = II;
end

ims1 = cat(4,ims{:});

X = fevalArrays(ims1,@(x)col(fhog(im2single(x))));

[IDX,C] = kmeans2(X',20,struct('nTrial',1,'display',1,'outFrac',.1));
maxPerCluster = .5;

% make the clustering consistent between different runs...
clustersPath = '~/mircs/experiments/common/faceClusters_big.mat';

% if (~exist(clustersPath,'file'))
    [clusters,ims_,inds] = makeClusterImages(ims,C',IDX',X,resultDir,maxPerCluster);
    conf.features.winsize = [10 10];
%     c_trained = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'suffix','face_20');
    % need only clusters 1,2,3 since 4,5 are mirrors of previous ones (!!)
%     c_ = [1 2 3];    
    save(clustersPath,'clusters','IDX','C');
%     load(clustersPath);
%     [clusters,ims_,inds] = makeClusterImages(ims,C',IDX',X,resultDir,maxPerCluster);
% end
save('~/mircs/experiments/common/faceClusters_big_ims.mat','ims');
% clusters = clusters(c_);
% ims_ = ims_(c_);
% inds = inds(c_);
partModelsDPM = {};
nonPersonIDS = getNonPersonIds(VOCopts);

% make it quick the first time just to see some results...
for iCluster = 1:length(clusters)
    curInds = inds{iCluster};    
    
    % make the paths...
    posImages = {};
    for q = 1:length(curInds)
        q
        posImages{q} = imread(fullfile(baseDir, pos_train_images(curInds(q)).name));
%         posImages{q} = imResample(posImages{q},.5,'bilinear');
%         clf; imagesc(posImages{q}); axis image; drawnow
    end
    
    trainSet = prepareForDPM(conf,posImages,nonPersonIDS(1:end),[]);    
    n = 1; % number of subclasses
    valSet = [];
    cls = ['face_big_' num2str(iCluster)];
    partModelsDPM{iCluster} = runDPMLearning(cls, n, trainSet, valSet);
end


