addpath('/home/amirro/code/3rdparty/vlfeat-0.9.14/toolbox');
addpath(genpath('/home/amirro/code/3rdparty/libsvm-3.12'));
addpath('/home/amirro/code/3rdparty/ssim');
addpath('/home/amirro/data/VOCdevkit/VOCcode/');
addpath('/home/amirro/code/3rdparty/voc-release4.01');
addpath('/home/amirro/code/fragments');
addpath('~/code/3rdparty/uri/');

vl_setup;

% vl_setup;
% addpath(genpath('D:\libsvm-3.12'));

opts.hands_locs_suff = 'hands_locs';
opts.hands_images_suff = 'hands_imgs';

% uncomment the following line if you wish to run the labeling tool.
%labeling_script;

% change this directory to where you put your standford40 dataset
%inputDir = 'D:\Stanford40\';
inputDir = '/home/amirro/data/Stanford40/JPEGImages';

ext = '.jpg';

actionsFileName = '/home/amirro/data/Stanford40/ImageSplits/actions.txt';
[A,ii] = textread(actionsFileName,'%s %s');

f = fopen(actionsFileName);
A = A(2:end);

% 3 -> brushing teeth
% 9 -> drinking
% 24 -> phoning
% 40 -> writing on a book
% 31 -> taking a photo
% 32 -> texting message

globals;

[train_annotations,test_annotations] = getTrainData('_hands_imgs_large4');
load('1000_10e4_vocab');
pos_samples = [];
neg_samples = [];


% calculate bow-images for all...

globalOpts.numWords = 1000;
load('1000_10e4_vocab');

kdtree = vl_kdtreebuild(vocab);

sizes = 4;
train_quants = {};


for k = 1:length(train_annotations)
    k
    c = train_annotations(k).imgname;
    im = im2single(imread(c));
    [F,D] = vl_phow(im,'Step',1,'Sizes',sizes,'Fast',1);
    F = round(F);
    quantized = double(vl_kdtreequery(kdtree, vocab, single(D),...
        'MaxComparisons', 15)) ; %#ok<NASGU>
    sz = size(im);
    sz = sz(1:2);
    [quantImage] = im2QuantImage( F,quantized,sz,globalOpts);
    train_quants{k} = quantImage;
    
end


test_quants = {};

for k = 1:length(test_annotations)
    k
    c = test_annotations(k).imgname;
    im = im2single(imread(c));
    [F,D] = vl_phow(im,'Step',1,'Sizes',sizes,'Fast',1);
    F = round(F);
    quantized = double(vl_kdtreequery(kdtree, vocab, single(D),...
        'MaxComparisons', 15)) ; %#ok<NASGU>
    sz = size(im);
    sz = sz(1:2);
    [quantImage] = im2QuantImage( F,quantized,sz,globalOpts);
    test_quants{k} = quantImage;    
end

%%

cls = classes{1}

globalOpts.numSpatialX = [1 2];
globalOpts.numSpatialY = [1 2];

pos = false(size(train_annotations));

for k = 1:length(train_annotations)
    if (strcmp(train_annotations(k).objects(1).class,cls))
        pos(k) = 1;
    end
end

all_samples = [];
for k = 1:length(train_annotations)
    if (mod(k,50)==0)
        disp(k);
    end
    q = train_quants{k};
    sz = size(q);
    hist_ = buildSpatialHist2(q,[1 1 fliplr(sz)],globalOpts);
    all_samples = [all_samples,hist_];
%     imagesc(q);
%     pause;
end

pos_ids = find(pos);
neg_ids = find(~pos);

nPosSamples = length(pos_ids);
nNegSamples = length(neg_ids);
pos_feats = all_samples(:,pos_ids);
neg_feats = all_samples(:,neg_ids);
% pos_feats = repmat(pos_feats,1,round(nNegSamples/nPosSamples));
% nPosSamples = size(pos_feats,2);

y = [ones(1,nPosSamples),...
    -ones(1,nNegSamples)];

psix = hkm([pos_feats,neg_feats]);

p = Pegasos(psix', int8(y)','iterNum', 10000);%,'lambda',9.05e-05);

w = real(p.w);
model.b = w(end, :) ;
model.w = w(1:end-1, :);


test_samples = [];
for k = 1:length(test_annotations)
     if (mod(k,50)==0)
        disp(k);
    end
    q = test_quants{k};
    sz = size(q);
    hist = buildSpatialHist2(q,[1 1 fliplr(sz)],globalOpts);
    test_samples = [test_samples,hist];
end

pos = false(size(test_annotations));

for k = 1:length(test_annotations)
    if (strcmp(test_annotations(k).objects(1).class,cls))
        pos(k) = 1;
    end
end


pos_ids = find(pos);
neg_ids = find(~pos);

nPosSamples = length(pos_ids);
nNegSamples = length(neg_ids);
pos_feats = test_samples(:,pos_ids);
neg_feats = test_samples(:,neg_ids);
% 
psix_test = hkm([pos_feats,neg_feats]);

scores = model.w' * psix_test + (model.b' * ones(1,size(psix_test,2)));

figure,plot(scores);

% scores = scores-min(scores);
% scores = scores/max(scores);

[ss,iss] = sort(scores,'descend');

tp = pos(iss);
fp = ~pos(iss);

fp=cumsum(fp);
tp=cumsum(tp);
rec=tp/sum(pos);
prec=tp./(fp+tp);
ap=VOCap(rec,prec);

plot(rec,prec,'-');
grid;
xlabel 'recall'
ylabel 'precision'
title_string = sprintf('class: %s, AP = %.3f',cls,ap);
 title(title_string);
% 
% [tpr,fpr,thresholds] = roc(y>0,scores)
% 
% plot(fpr,tpr);

