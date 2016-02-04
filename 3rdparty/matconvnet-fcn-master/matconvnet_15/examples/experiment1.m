expDir = '/home/amirro/storage/data/imagenet12-vgg-f-bnorm-simplenn-ratio_10';
imdb = load(fullfile(expDir,'imdb.mat'));

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'))
addpath(genpath('~/code/utils'))
imageStatsPath = fullfile(expDir,'imageStats.mat');
load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
[v,d] = eig(rgbCovariance) ;
bopts.transformation = 'stretch' ;
bopts.averageImage = rgbMean ;
bopts.rgbVariance = 0.1*sqrt(d)*v' ;

% validation: 50,000
% test : 100,000
test_set = imdb.images.set==2; % validation set 
test_images = imdb.images.name(test_set);
test_images = fullfile(imdb.imageDir,test_images);

test_labels = imdb.images.label(test_set);

[v,iv] = sort(test_labels,'ascend');

test_labels_sub = test_labels(iv(1:1:end));
test_images_sub = test_images(iv(1:1:end));

img_batches = {};
batches = batchify(length(test_images_sub),length(test_images_sub)/200);
for iBatch = 1:length(batches)
    iBatch
    img_batches{iBatch} = test_images_sub(batches{iBatch});
end
%     vl_imreadjpeg(img_batches{iBatch},'NumThreads',12,'Prefetch','Preallocate',true);
%     imgs = vl_imreadjpeg(img_batches{iBatch},'NumThreads',12);
%     for t = 1:length(imgs)
%         if size(imgs{t},3)>3
%             t
%             warning('found more than 3 channels in input image...');
%             imgs{t} = imgs{t}(:,:,1:3);
%         end
%         imgs{t} = uint8(imgs{t});
%     end
%     img_batches{iBatch} = imgs;
% end

% vl_imreadjpeg(test_images_sub,'NumThreads',12,'Prefetch','Preallocate',false);
% actual_test_images = vl_imreadjpeg(test_images_sub,'NumThreads',12);
% 
% for t = 1:length(actual_test_images)
%     if size(actual_test_images{t},3)>3
%         t
%         warning('found more than 3 channels in input image...');
%         actual_test_images{t} = actual_test_images{t}(:,:,1:3);
%     end
% end

% imo = cnn_imagenet_get_batch(actual_test_images,bopts);
imo = {};
for t = 1:length(img_batches)
    t
    M = img_batches{t};
    for tt = 1:length(M)
        M{tt} = single(M{tt});
    end
    imo{t} = cnn_imagenet_get_batch(M,bopts);
end

all_cms = {};
for ii = 1:20
        ii
    load(sprintf([expDir '/net-epoch-%d.mat'],ii));
    net = vl_simplenn_move(net, 'gpu') ;    
    net.layers = net.layers(1:end-1); 
%     profile on
    [cm,X] = calcPerf2(net,imo,test_labels_sub,bopts);
%     profile off
%     test_images = imdb.images.data(:,:,:,imdb.images.set==3);
%     test_labels = imdb.images.labels(imdb.images.set==3);
    all_cms{ii} = cm;
end

% save ~/storage/misc/all_cms.mat all_cms

%%
diags = {};
for t = 1:length(all_cms)
    diags{t} = diag(all_cms{t});
end
diags = cat(2,diags{:});

% diags: N_classes x N_epochs

%plot(diags')
bestPerf = max(diags,[],2); 
finalPerf = diags(:,end);
figure,plot(finalPerf,bestPerf,'r+')
figure,plot(max(diags,[],2)./diags(:,end))

figure,hist(bestPerf-finalPerf)
mean(bestPerf)
mean(finalPerf)
hist(bestPerf)

% figure,hist(bestPerf./(finalPerf+eps),1:.5:10)

% figure,plot(diags(96,:));

% Interesting, the best attained performance can be much better than the
% final one...
mean(diags)
figure(1); clf; hold on; plot(diags(:,end),'g-');
hold on;
plot(max(diags,[],2),'b-');
figure,stem(bestPerf-finalPerf)

goods = bestPerf > 0;
hist( (bestPerf-finalPerf)./(bestPerf+eps),15)

[r,ir] = max( (bestPerf-finalPerf)./(bestPerf+eps));
bestPerf(ir)
finalPerf(ir)
imagesc(all_cms{end})

% find the most correlated classes.


%% 
