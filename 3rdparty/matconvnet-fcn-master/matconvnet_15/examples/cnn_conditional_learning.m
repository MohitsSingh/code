% function [net, info] = cnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST
% addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox/')
% vl_setup
% run(fullfile(fileparts(mfilename('fullpath')),   '..', 'matlab', 'vl_setupnn.m')) ;
% 
% 
% addpath(genpath('~/code/utils'));

opts.expDir = fullfile('data','conditional') ;
opts.dataDir = fullfile('data','mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.useBnorm = false ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 5 ;
opts.train.continue = true ;
opts.train.printToPdf = false;
opts.train.gpus = 1;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
imdbPath = 'data/conditional/imdb.mat';
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    
    imdb = getChupchik_IMDB();
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end
opts.useBnorm = false;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.numEpochs = 20;
opts.train.batchSize = 20;
opts.train.skipSaving = false;
opts.train.errorFunction = 'attributes';
opts.train.learningRate = .001;%[ones(1,10)*.001,ones(1,10)*.0001,ones(1,10)*.00001];
% opts.train.errorFunction = 'multiclass';
% profile off

partial_results = struct('nEpochs',{},'net',{},'x',{},'ap1',{},'ap2',{});
test_set = imdb.images.set==2;
test_data = imdb.images.data(:,:,:,test_set);
test_labels = squeeze(imdb.images.labels(:,:,:,test_set));
net = cnn_chupchick_init('useBnorm', opts.useBnorm) ;
opts.train.loss = 'logistic';
[net, info] = cnn_train(net, imdb, @getBatch, opts.train,'val', find(imdb.images.set == 3)) ;
for iNumEpochs = 1:opts.train.numEpochs
    %net = cnn_chupchick_init('useBnorm', opts.useBnorm) ;
    opts.train.loss = 'logistic';
    %[net, info] = cnn_train(net, imdb, @getBatch, opts.train,'val', find(imdb.images.set == 3)) ;
    load(sprintf('data/conditional/net-epoch-%d.mat',iNumEpochs));
    net.layers = net.layers(1:end-1);
    res = vl_simplenn(net,test_data);
    x = squeeze(res(end).x);
    [recall,precision,info] = vl_pr( test_labels(1,:),x(1,:));
    partial_results(iNumEpochs).nEpochs = iNumEpochs;
    partial_results(iNumEpochs).net = net;
    partial_results(iNumEpochs).ap1 = info.ap;
    [recall,precision,info] = vl_pr( test_labels(2,:),x(2,:));
    partial_results(iNumEpochs).ap2 = info.ap;
end
%%
figure(1);
task_ = 2;
[r,ir] = sort(x(task_,:),'descend');
for it = 1:length(ir)
    fprintf('.')
    clf; imagesc2(squeeze(test_data(:,:,:,ir(it))));
    %     r(it)
    dpc(.5)
end
% %%
% figure(1),clf,vl_pr( test_labels(1,:),x(1,:))
% figure(2),clf,vl_pr( test_labels(2,:),x(2,:))
figure(100);clf; plot([partial_results.ap1]);
hold on; plot([partial_results.ap2]);
ylim([0 1.5]);
%%%%%%%%
%%%%%%%%
%%%%%%%
%% Do the exact same but without the help of the big square
opts.expDir = fullfile('data','conditional_nohelp') ;
opts.dataDir = fullfile('data','mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.expDir = opts.expDir ;
imdb = load('data/conditional/imdb.mat');
imdb.images.labels(:,:,1,:) = -1;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.skipSaving = false;
net = cnn_chupchick_init('useBnorm', opts.useBnorm) ;
opts.train.loss = 'logistic';
[net, info] = cnn_train(net, imdb, @getBatch, opts.train,'val', find(imdb.images.set == 3)) ;

net.layers = net.layers(1:end-1);
partial_results2 = struct('nEpochs',{},'net',{},'x',{},'ap1',{},'ap2',{});
net = cnn_chupchick_init('useBnorm', opts.useBnorm) ;
opts.train.loss = 'logistic';
[net, info] = cnn_train(net, imdb, @getBatch, opts.train,'val', find(imdb.images.set == 3)) ;
for iNumEpochs = 1:opts.train.numEpochs
    %net = cnn_chupchick_init('useBnorm', opts.useBnorm) ;
    opts.train.loss = 'logistic';
    %[net, info] = cnn_train(net, imdb, @getBatch, opts.train,'val', find(imdb.images.set == 3)) ;
    load(sprintf('data/conditional_nohelp/net-epoch-%d.mat',iNumEpochs));
    net.layers = net.layers(1:end-1);
    res = vl_simplenn(net,test_data);
    x = squeeze(res(end).x);
    [recall,precision,info] = vl_pr( test_labels(2,:),x(2,:));
    partial_results2(iNumEpochs).nEpochs = iNumEpochs;
    partial_results2(iNumEpochs).net = net;
    partial_results2(iNumEpochs).ap2 = info.ap;
    
end
% res = vl_simplenn(net,test_data);
% x = squeeze(res(end).x);
% % 
% figure(2);clf; 
figure(100); hold on; plot([partial_results2.ap2],'g--');
legend('task1','task2','task2_nohelp');



%%
% figure(100);
% task_ = 2;
% [r,ir] = sort(x(task_,:),'descend');
% for it = 1:length(ir)
%     fprintf('.')
%     clf; imagesc2(squeeze(test_data(:,:,:,ir(it))));
%     %     r(it)
%     dpc(.5)
% end
% %%
% figure(1),clf,vl_pr( test_labels(1,:),x(1,:))
% figure(2),clf,vl_pr( test_labels(2,:),x(2,:))

