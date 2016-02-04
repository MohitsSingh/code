function imgs = alignByJittering(conf,images,visualize)
% jitters = {};

%images = cat(4,cupImages{:});
% if (0)
% images = multiRead(conf,'/home/amirro/storage/data/action_drink/imgs');

% get features of all images.
conf.features.vlfeat.cellsize = 8;

sz = [64 64];
X = imageSetFeatures2(conf,images,true,sz);
images = cat(4,images{:});
conf.features.winsize = 1*[8 8]*sz(1)/64;
D = l2(X',X');

learnModelsDPM

distFun = @(x) exp(-x/100);
curEnergy = sum(distFun(D).*(1-eye(size(D))),2);
imgs = im2double(images);
nIterations = 200;
energies = zeros(1,nIterations);
for i=1:length(imgs); labels{i}=[int2str2(i,2)]; end


% prm = struct('nPhi',10,'mPhi',10,'maxn',20,'hasChn',1,'flip',1,...
%         'mTrn',4,'nTrn',5,'scls',[1 1; 1.2 1; 1 1.3; 1.3 1.3]);
r = randperm(size(images,4));
% end
%
prms ={};

% prm1 = struct('hasChn',1,'flip',1); % flips
% prm2 = struct('hasChn',1,'mPhi',40,'nPhi',10); % small rotations
% prm3 = struct('hasChn',1, 'mTrn',4,'nTrn',5); % small translations
% prm4 = struct('nPhi',0,'mPhi',0,'maxn',10,'hasChn',1,'flip',1,...
%     'mTrn',4,'nTrn',5);
prm4 = struct('nPhi',10,'mPhi',15,'hasChn',1,'flip',1,...
    'mTrn',4,'nTrn',5,'maxn',10);
% prm4 = struct('maxn',10,'hasChn',1,'flip',0,...
%         'mTrn',4,'nTrn',5,'scls',[1 1; 1.2 1; 1 1.3; 1.3 1.3]);


% prms = {prm1,prm2,prm3};
prms = {prm4};

nChange = 100;
profile on;
for k = 1:nIterations
    
    % pick the most violating image.
    %         [m,iImage] = m(curEnergy);
    iImage = r(mod(k,size(images,4))+1);
    
    energies(k) = sum(curEnergy);
    
    k
    mm = 2;
    nn = 1;
    if (visualize && mod(k,50)==0)
        clf; subplot(mm,nn,1);
        
        
        MM = mean(imgs,4);
        imagesc(MM); axis image;
        
%         montage2(imgs,struct('hasChn',1,'labels',{labels}));
        title(num2str(sum(curEnergy)));
        if (k > 1)
            subplot(mm,nn,2); plot(energies(1:k));
        end
%         subplot(mm,nn,3); imagesc(showHOG(conf,mean(X,2).^2));axis image;
        
        fprintf(1,'total energy: %05.3f\n',sum(curEnergy));
        pause(.1);
    end
    I = squeeze(images(:,:,:,iImage));
    kk = ceil(k/nChange);
    curPrm = prms{min(length(prms),kk)};
    IJ = jitterImage(I,curPrm);
    M = mat2cell2(IJ,[1 1 1 size(IJ,4)]);
    X_ = imageSetFeatures2(conf,M,true,sz);
    curInds = setdiff(1:size(images,4),iImage);
    
    
    
    curDists = sum(distFun(l2(X(:,curInds)',X_')));
    [m,im] = max(curDists);
    imgs(:,:,:,iImage) = im2double(IJ(:,:,:,im));
    X(:,iImage) = X_(:,im);
    
    D = l2(X',X');
    %curEnergy = sum(distFun(D),2);
    curEnergy = sum(distFun(D).*(1-eye(size(D))),2);
end


% % % % %%
% % % % figure,montage2(images,struct('hasChn',1,'labels',{labels}));title('before');
% % % % figure,montage2(imgs,struct('hasChn',1,'labels',{labels}));title('after');
% end
%


%% now learn a model....
% 
% clusters = makeCluster(double(X),[]);
% clusters_trained = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'suffix','cups_1','override',true);
% 
% 
% figure,imshow(showHOG(conf,clusters_trained));
% [test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
% qq = applyToSet(conf,clusters_trained,test_ids(test_labels),[],'cup_top_check','override',true,'disp_model',true,...
%     'uniqueImages',true);
% 
% 
% imshow('cup_top_check.jpg')