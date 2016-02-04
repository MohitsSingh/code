% function [net, info] = my_cnn_cifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.
if (0)
    addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox/')
    vl_setup
    run(fullfile(fileparts(mfilename('fullpath')), ...
      '..', 'matlab', 'vl_setupnn.m')) ;
    addpath(genpath('~/code/utils'));
    addpath(genpath('~/code/3rdparty/piotr_toolbox'));
end


gpuDevice(2);
opts.modelType = 'lenet' ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

switch opts.modelType
    case 'lenet'
        opts.train.learningRate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
        opts.train.weightDecay = 0.0001 ;
    case 'nin'
        opts.train.learningRate = [0.5*ones(1,30) 0.1*ones(1,10) 0.02*ones(1,10)] ;
        opts.train.weightDecay = 0.0005 ;
    otherwise
        error('Unknown model type %s', opts.modelType) ;
end
opts.expDir = fullfile('data', sprintf('cifar-100-%s', opts.modelType)) ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train.batchSize = 100 ;
opts.train.continue = true ;
opts.train.gpus = 2 ;
opts.train.expDir = opts.expDir ;
% opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

switch opts.modelType
    case 'lenet', net = cnn_cifar_init(opts) ;
    case 'nin',   net = cnn_cifar_init_nin(opts) ;
end

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    %
    clear imdb;
    meta = load('/home/amirro/storage/data/cifar-100-matlab/meta.mat');
    train = load('/home/amirro/storage/data/cifar-100-matlab/train.mat');
    test = load('/home/amirro/storage/data/cifar-100-matlab/test.mat');
    train.data = reshape(train.data',32,32,3,[]);
    test.data = reshape(test.data',32,32,3,[]);

    sets = 1*ones(size(train.fine_labels));
    sets(end*3/4:end) = 3;
    sets = [sets;2*ones(size(test.fine_labels))]';
    
    data = cat(4,train.data,test.data);
    data = permute(data,[2 1 3 4]);
    fine_labels = [train.fine_labels;test.fine_labels]';
    coarse_labels = [train.coarse_labels;test.coarse_labels]';
    
    dataMean = mean(data(:,:,:,sets == 1), 4);
    data = bsxfun(@minus, single(data), dataMean);

    imdb.images.data = data(:,:,:,1:1:end);
    imdb.images.labels = single(fine_labels(1:1:end)+1);
    imdb.images.set = sets(1:1:end);
        

%     for t = 1:size(train.data,1)
% %         I = reshape(train.data(t,:),32,32,3);
%         clf; imagesc2(train.data(:,:,:,t));
%         pause
%     end

%     imdb = getCifarImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.gpus=1;
[net, info] = cnn_train(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
net = vl_simplenn_move(net, 'gpu') ;

all_cms = {};
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
test_labels = imdb.images.labels(imdb.images.set==3);
all_results = {};
for ii = 1:30
        ii
    load(sprintf('data/cifar-100-lenet/net-epoch-%d.mat',ii));
    net = vl_simplenn_move(net, 'gpu') ;    
    net.layers = net.layers(1:end-1);        
%     test_images = test_images;
%     test_labels = test_labels(1:500);
    [cm,all_results{ii}] = calcPerf(net,test_images,test_labels,1);
%     test_images = imdb.images.data(:,:,:,imdb.images.set==3);
%     test_labels = imdb.images.labels(imdb.images.set==3);
    all_cms{ii} = cm;
end

aucs = zeros(size(diags));
for t = 1:length(all_results)
    t
    curResults = all_results{t};
     E = exp(bsxfun(@minus, curResults, max(curResults,[],1))) ;
    L = sum(E,1) ;
    curResults = bsxfun(@rdivide, E, L) ;
    for tt = 1:100
        [prec,rec,info] = vl_pr(2*(test_labels==tt)-1,curResults(tt,:));
        aucs(tt,t) = info.ap;
    end    
end

figure,plot(aucs')
%%
diags = {};
for t = 1:length(all_cms)
    diags{t} = diag(all_cms{t});
end
diags = cat(2,diags{:});
plot(diags')
% 
% 
% all_aps_val = {};
% val
% for ii = 1:30
%         ii
%     load(sprintf('data/cifar-100-nin/net-epoch-%d.mat',ii));
%     net = vl_simplenn_move(net, 'gpu') ;    
%     [aps,X] = calcPerf(net,test_images,test_labels,10);
%     all_aps_val{ii} = aps;
% end
diags = aucs;
bestPerf = max(diags,[],2);
finalPerf = diags(:,end);
figure,plot(finalPerf,bestPerf,'r+')
figure,plot(max(diags,[],2)./diags(:,end))
% figure,plot(diags(96,:));

% Interesting, the best attained performance can be much better than the
% final one...
mean(diags)
figure(1); clf; hold on; plot(diags(:,end),'g-');
plot(max(diags,[],2),'b-');
figure,stem(bestPerf-finalPerf)
mean(bestPerf-finalPerf)

hist(iv,1:30)

figure,plot(mean(diags,1));

%% train again with a better initialization
train_images = find(imdb.images.set==1);
opts.modelType = 'lenet' ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

switch opts.modelType
    case 'lenet'
        opts.train.learningRate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
        opts.train.weightDecay = 0.0001 ;
    case 'nin'
        opts.train.learningRate = [0.5*ones(1,30) 0.1*ones(1,10) 0.02*ones(1,10)] ;
        opts.train.weightDecay = 0.0005 ;
    otherwise
        error('Unknown model type %s', opts.modelType) ;
end

switch opts.modelType
    case 'lenet', net_pca = cnn_cifar_init(opts) ;
    case 'nin',   net_pca = cnn_cifar_init_nin(opts) ;
end

% test_images = imdb.images.data(:,:,:,imdb.images.set==3);
tr = train_images(randperm(length(train_images)));
tr = tr(1:10000);
[y,x] = meshgrid(1:5:27,1:5:27);
tr = imdb.images.data(:,:,:,tr);
tr_ = {};
locs = randint2(1,size(tr,4),[1 length(x(:))]);
for t = 1:length(locs)
    m = locs(t);
    p = tr(x(m):x(m)+4,y(m):y(m)+4,:,t);
    tr_{t} =p;
end
tr_ = cat(4,tr_{:});
tr_ = reshape(tr_,[],size(tr_,4));
[ U, mu, vars ] = pca( tr_ );
U = reshape(U(:,1:32),5,5,3,[]);
net_pca.layers{1}.weights{1} = single(U);
%%
opts.train.expDir = 'data/cifar-100-lenet_with_pca';
[net_pca, info] = cnn_train(net_pca, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));
%%
all_cms_pca = {};
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
test_labels = imdb.images.labels(imdb.images.set==3);
all_results_pca = {};
for ii = 1:30
        ii
    load(sprintf('data/cifar-100-lenet_with_pca/net-epoch-%d.mat',ii));
    net = vl_simplenn_move(net, 'gpu') ;    
    net.layers = net.layers(1:end-1);        
%     test_images = test_images;
%     test_labels = test_labels(1:500);
    [cm,all_results_pca{ii}] = calcPerf(net,test_images,test_labels,1);
%     test_images = imdb.images.data(:,:,:,imdb.images.set==3);
%     test_labels = imdb.images.labels(imdb.images.set==3);
    all_cms_pca{ii} = cm;
end

figure(1); clf;imagesc(all_cms_pca{end}); title('pca');
figure(2); clf; imagesc(all_cms{end}); title('no pca');
mean(diag(all_cms_pca{end}));
mean(diag(all_cms{end}));

%% again, with gabor filters.

% make a net with gabor filters all the way.
net_gabor = cnn_cifar_init(opts);
for t = 1:length(net_gabor.layers)
    curLayer = net_gabor.layers{t}; 
    if strcmp(curLayer.type,'conv')
        nFilts = size(curLayer.weights{1},4);
        rad = floor(size(curLayer.weights{1},1)/2);
        newFilts = FbMakegabor( rad, nFilts/2, 1, 2, 2 );
                        
        if t==1
            newFilts = reshape(newFilts,5,5,1,32);
            newFilts = repmat(newFilts,1,1,3,1);
            sz = size(newFilts);
            newFilts = reshape(normalize_vec(reshape(newFilts,[],sz(end))),sz);
            
%             newFilts = bsxfun(@rdivide,newFilts,sum(sum(sum(newFilts.^2,1),2),3).^.5);
        end        
        size(curLayer.weights{1})
        curLayer.weights{1} = single(newFilts);
        net_gabor.layers{t} = curLayer;
    end
    break
end

% % FB = FbMakegabor( rad, nFilts/2, 1, 2, 2 );
% % FbVisualize( FB, 1 );
% % 

opts.train.expDir = 'data/cifar-100-lenet_with_gabor';
[net_gabor, info] = cnn_train(net_gabor, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));

%%
for t = 1:30
    figure(1);clf; imagesc2((all_cms{t}));dpc(.1)
end

figure(2),vl_pr(2*(test_labels==1)-1,all_results{end}(1,:))

figure,imagesc(all_cms{end})


%% score each class according to it's best epoch

[v,iv] = max(diags,[],2);
newX = zeros(size(all_results{1}));
% newX = all_results{end};
% E = exp(bsxfun(@minus, newX, max(curResults,[],1))) ;
% L = sum(E,1) ;
% newX = bsxfun(@rdivide, E, L) ;

for t = 1:length(iv)
    curResults = all_results{iv(t)};
    % do a softmax...
    E = exp(bsxfun(@minus, curResults, max(curResults,[],1))) ;
    L = sum(E,1) ;
    Y = bsxfun(@rdivide, E, L) ;
    newX(t,:) = .5*Y(t,:);
%     [m,im] = max(curResults,[],1);
%     newX(t,im==t) = 1;
%     newX(t,:) = newX(t,:)+.5*(im==t);%.1*Y(t,:);
end


% newX = mean(cat(3,all_results{:}),3);

[v,iv] = max(newX,[],1);
nClasses = size(newX,1);
cm = confMatrix(test_labels,single(iv),nClasses);%,varargin)
cm = bsxfun(@rdivide,cm,sum(cm,2)+eps);
mean(diag(cm))

figure(1),subplot(1,2,1);
imagesc(cm); title(sprintf('after : %f',mean(diag(cm))));
subplot(1,2,2),imagesc(all_cms{end}); title(sprintf('before : %f',mean(diag(all_cms{end}))));

%%
[v,iv] = max(all_results{end},[],1);
cm = confMatrix(test_labels,single(iv),nClasses);%,varargin)

figure,imagesc(cm)

cm(1)/sum(cm(1,:))
cm(1)/sum(cm(:,1))


%%
A = diags';
size(A)
mean(A(end,:))
% A = cat(2,all_cms{:})';
% A = smooth(all_aps);
% for t = 1:30
%     A(t,:) = smooth(A(t,:),5,'rlowess');
% end
% show all tasks relative to the maximal.
A_orig=A;
% clf; plot(mean(A,2))
A = bsxfun(@rdivide,A,max(A,[],1));
clf; plot(A);drawnow
A = diags';
% aucs = zeros(10fi0,1);
for t = 1:100
    r = 1:30;
    p = A(:,t);
    aucs(t) = trapz(r,p);
end

[v,iv] = sort(aucs,'descend');
for it = 1:size(A,2)
t = iv(it);
 clf; plot(A(:,t)); title(sprintf('%d , %s',t,meta.fine_label_names{t}));
dpc
% now
end

%%
A

%lets see if there is a correlation between the speed of learning and final
%accuracy.
final_ap = A_orig(end,:);
[max_ap,imax] = max(A_orig,[],1);
figure,plot(aucs,final_ap,'r+');xlabel('auc'),ylabel('final ap');

figure,plot(aucs,max_ap,'r+');xlabel('auc'),ylabel('max ap');
figure,plot(max_ap,final_ap,'r+')
figure,hist(imax)
figure,plot(imax,final_ap,'r+')

%%
% train on a single class
wanted_classes = 1;
other_class = 10;
imdb_sub = restrict_to_sub(imdb,wanted_classes,other_class);

switch opts.modelType
    case 'lenet', net = cnn_cifar_init(opts) ;
    case 'nin',   net = cnn_cifar_init_nin(opts) ;
end
opts.expDir = fullfile('data','cifar-baseline-subset') ;
opts.train.expDir = opts.expDir ;


[net, info] = cnn_train(net, imdb_sub, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

%%
% --------------------------------------------------------------------

% now check the mean average precision
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
net.layers = net.layers(1:end-1);
test_res = vl_simplenn(net,test_images);
X = squeeze(test_res(end).x);
size(X)
test_labels = imdb.images.labels(imdb.images.set==3);
aps = zeros(10,1);
for t = 1:10
    [prec,rec,info] = vl_pr(2*(test_labels==t)-1,X(t,:));
    aps(t) = info.ap;
end

%%


%%

% now make a new random network.
net_from_random = trainRandomCIFARArchitecture(50);
% ok, now learn only the first label.
opts.expDir = fullfile('data','cifar-baseline-subset') ;
opts.train.expDir = opts.expDir ;
myopts = struct('useBnorm',opts.useBnorm,'subset',2);
net_partial = cnn_cifar_init(myopts);
imdb.images.labels(imdb.images.labels~=1) = 2;
[net_partial, info] = cnn_train(net_partial, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% test this net on the first label
net_partial.layers = net_partial.layers(1:end-1);
test_res = vl_simplenn(net_partial,test_images);
X = squeeze(test_res(end).x);
size(X)
test_labels = imdb.images.labels(imdb.images.set==3);
aps = zeros(2,1);
for t = 1:2
    [prec,rec,info] = vl_pr(2*(test_labels==t)-1,X(t,:));
    aps(t) = info.ap;
end

% --------------------------------------------------------------------

