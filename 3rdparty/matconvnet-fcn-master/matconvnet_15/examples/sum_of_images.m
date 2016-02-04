% function cnn_imagenet_minimal()
% CNN_IMAGENET_MINIMAL   Minimalistic demonstration of how to run an ImageNet CNN model

% setup toolbox
% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', 'matlab', 'vl_setupnn.m')) ;

% download a pre-trained CNN from the web
% if ~exist('imagenet-vgg-f.mat', 'file')
%   urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
%     'imagenet-vgg-f.mat') ;
% end
% net = load('/home/amirro/storage/matconv_data/imagenet-vgg-f.mat') ;
% net = vl_simplenn_move(net, 'gpu') ;
% obtain and preprocess an image
% im = imread('peppers.png') ;
% im_ = single(im) ; % note: 255 range
% im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
% im_ = im_ - net.normalization.averageImage;
%I = imread('peppers.png');
%%
sz = 128;
I1 = imread('/home/amirro/cat1.jpg');

I1 = imResample(I1,[sz sz]);

% I1=  imResample(I1,[224 224]);
[scores1,bestScore1,best1] = predictAndShow(I1,net,1);
ent(scores1)
%%
I2 = imread('~/dog.jpg');
I2=  imResample(I2,[sz sz]);
[scores2,bestScore2,best2] = predictAndShow(I2,net,2);
ent(scores2)
%%
alpha_ = .5;
I3 = alpha_*I1 + (1-alpha_)*I2;
[scores3,bestScore3,best3] = predictAndShow(I3,net,3);
ent(scores3)
I3 = single(I3);
%%
% try to minimize the entropy.
prevEnt = ent(scores1);
img_orig = single(I1);
curImage = single(img_orig);
%%
N  =0;
% curImage = curImage+rand(size(curImage))*50;
ENTROPY = 1;
OTHER_OBJECT = 2;
target_object = 986; % daisy :-)
%target_object = 286;
target_object = 1;
% target_object = 1:100:1000;
goalType = OTHER_OBJECT;
if goalType==ENTROPY
    prevScore = prevEnt;
else
    z_target = zeros(size(scores,1),1);
    z_target(target_object) = 1;
%     z_target = z_target/norm(z_target);
%     prevScore = l2(normalize_vec(z_target)',scores1');
    prevScore = -z_target'*scores1;
end
%%
% prevEnt = inf;
% goalType = ENTROPY;
% if goalType==ENTROPY
%     prevScore = prevEnt;
% else
%     z_target = zeros(size(scores,1),1);
%     z_target(target_object) = 1;
%     prevScore = -z_target'*scores1;
% end
curTheta = 0;
 batchSize = 64;
z = randn([size(curImage) batchSize]);
for t = 1:1000
    t
    % do it in batches....
   
    
    I4 = bsxfun(@plus,curImage,z);
%     TT = curTheta+randn(1)*50;
%     I4 = imrotate(img_orig,TT,'bilinear','crop');
    [curScores,bestScore,best] = predictAndShow(I4,net,0);
    if goalType==ENTROPY
        scoreFunction = ent(curScores);
        [curScore,iv] = min(scoreFunction);
    else
%         R = l2(normalize_vec(z_target)',normalize_vec(curScores)');
%         [curScore,iv] = min(R);
        [curScore,iv] = max(z_target'*curScores);
        curScore = -curScore;
    end
    if curScore < prevScore
        curTheta = TT;
        prevScore = curScore
        curImage = squeeze(I4(:,:,:,iv));
%         curImage = curImage-min(curImage(:));
%         curImage = 255*(curImage/max(curImage(:)));
        N = N+1;
        if (mod(N,1)==0)
            curScore
            figure(4) ; clf ;
            subplot(1,3,1);imagesc2(curImage/255) ;title('current');
            title(sprintf('%s\n (%d)\n score %.3f',...
                net.classes.description{best(iv)}, best(iv), bestScore(iv))) ;
            subplot(1,3,3); bar(curScores(:,iv));
            subplot(1,3,2);imagesc2(img_orig/255) ; title('orig');
            dpc(.1)
        end
    end
    
end


