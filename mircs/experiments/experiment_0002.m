%%%%%% Experiment 2 %%%%%%%
% Oct. 24, 2013

% make a new face detector using DPM and many faces from AFLW.
% The result of this is models face_1, face_2, face_3 in the dpm directory.
% after having done this, I will run the face detectors on the s40 images
% again. 


function experiment_0002
initpath;
config;
resultDir = ['~/mircs/experiments/' mfilename];
ensuredir(resultDir);
conf.features.vlfeat.cellsize = 8;
% fHandle = @(x) col(single(vl_hog(imResample(im2single(x),[64 64]),conf.features.vlfeat.cellsize,'NumOrientations',9)));
% X = fevalImages(fHandle,{},'/home/amirro/storage/data/faces/aflw_cropped/aflw_cropped','face','jpg',1);
baseDir = '~/storage/train_faces/aflw_cropped_context/';
d = dir(fullfile(baseDir,'*.jpg'));

qq = 4;
% qq = 30;
pos_train_images = d(1:qq:end);
pos_val_images = d(2:qq:end);

ims = {};
for k = 1:length(pos_train_images)
    if (mod(k,100)==0)
        k
    end
    I = imread(fullfile(baseDir, pos_train_images(k).name));
%     I = I(round(.25*end:.75*end),round(.25*end:.75*end),:);
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

X = fevalArrays(ims1,@(x)col(hog(im2single(x))));
[IDX,C] = kmeans2(X',5,struct('nTrial',1));
maxPerCluster = 300;
[clusters,ims_,inds] = makeClusterImages(ims,C',IDX',X,resultDir,maxPerCluster);
c_ = [1 2 4];
clusters = clusters(c_);
ims_ = ims_(c_);
inds = inds(c_);
% need only clusters 1,2,3 since 4,5 are mirrors of previous ones (!!)
partModelsDPM = {};
nonPersonIDS = getNonPersonIds(VOCopts);
for iCluster = 1:length(clusters)
    curInds = inds{iCluster};    
    
    % make the paths...
    posImages = {};
    for q = 1:length(curInds)
        q
        posImages{q} = imread(fullfile(baseDir, pos_train_images(curInds(q)).name));
%         clf; imagesc(posImages{q}); axis image; drawnow
    end
    
    trainSet = prepareForDPM(conf,posImages,nonPersonIDS(1:2:end),[]);    
    n = 1; % number of subclasses
    valSet = [];
    cls = ['face_' num2str(iCluster)];
    partModelsDPM{iCluster} = runDPMLearning(cls, n, trainSet, valSet);
end





