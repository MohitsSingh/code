function [confusion,cm_normalized] = test_segmentation_accuracy(modelPath,imdb,train,val,test,...
    test_params)

labels = test_params.labels;
labels_to_block = test_params.labels_to_block;

% run matconvnet/matlab/vl_setupnn ;
% addpath matconvnet/examples ;
opts.modelPath = modelPath;

opts.nClasses = 0;

% train = find(imdb.isTrain);
% val = find(~imdb.isTrain);

imdb.nClasses = 3; %

opts.useGpu = 2;
gpuDevice(opts.useGpu);
opts.modelFamily = 'matconvnet' ;

[net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath, opts.modelFamily);
net.move('gpu');

addpath('~/code/3rdparty/subplot_tight');
% labels_full = {'none','face','hand','obj'};
%%

test_set = test_params.set;

if strcmp(test_set,'test')
    test_set = test;
elseif strcmp(test_set,'train')
    test_set = train;
elseif strcmp(test_set,'val')
    test_set = val;
else
    error(sprintf('unknown test set %s specified',test_set));
end

all_scores = {};
all_masks = {};
N = length(labels);
% figure(1000)
% confusion = [];
confusion = zeros(N);
all_inds = {};


%% EXPERIMENTAL
d = dir('~/storage/data/Stanford40/JPEGImages/*.jpg');
dd = fullfile('~/storage/data/Stanford40/JPEGImages/',{d.name});
%%

outDir = '~/storage/s40_context_seg';
for ii = 501:50:length(dd)
    ii
    
    curImagePath = dd{ii};
    curOutPath = j2m(outDir,curImagePath, '.mat');
    if (exist(curOutPath,'file'))
        continue
    end
    rgb_orig = imread(curImagePath);
    rgb = single(rgb_orig);   
%     rgb = imResample(rgb,100/size(rgb,1),'bilinear');
    [pred,scores] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,'prediction',rgb,1,true,labels);
%     dpc
    dpc;continue            
    [scores,pred] = applyNet(net,rgb,imageNeedsToBeMultiple,inputVar,'prediction');
    scores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
%     scores = im2uint8(scores);
%     save(curOutPath,'scores');
    continue
            
    disp(nnz(pred(:)>1))
    %     for u = length(f):-1:1
    %         pred(pred==1+u) = f(u)+1;
    %     end
%     softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
    %   [~,pred] = max(scores,[],3) ;
    ok = lb > 0 ;
    %     if isempty(confusion)
    %         N = size(softmaxScores,3);
    %         confusion = zeros(N);
    %     end
    confusion = confusion + accumarray([lb(ok),pred(ok)],1,[N N]) ;
    if (mod(ii,1)==0)
        clf; imagesc2(bsxfun(@rdivide,confusion,sum(confusion,2)));
        title(sprintf('%d / %d, %f', ii, length(test_set), sum(diag(confusion))/sum(confusion(:))));
        drawnow
    end

end
%%

for ii = 1:1:length(test_set)
    ii
    imId = test_set(ii);
    rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});
    %rgb = imResample(rgb,2*[384 384],'bilinear');
    rgb_orig = rgb;
    sz_orig = size2(rgb);
    tic
    
    lb = single(imdb.labels{imId});
    
    % block out the required labels.
    toBlock = ismember(lb,labels_to_block);
    toBlock = repmat(toBlock,1,1,3);
    z = zeros(size(rgb));
    z(:,:,1) = net.meta.normalization.rgbMean(1);
    z(:,:,2) = net.meta.normalization.rgbMean(2);
    z(:,:,3) = net.meta.normalization.rgbMean(3);
    rgb(toBlock) = z(toBlock);
    lb = mod(lb + 1, 256) ; % 0 = ignore, 1 = bkg
%         [scores,pred] = applyNet(net,rgb,imageNeedsToBeMultiple,inputVar,'prediction');
        [pred,scores] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,'prediction',rgb,1,true,{'none','obj');
    dpc;continue
    [scores,pred] = applyNet(net,rgb,imageNeedsToBeMultiple,inputVar,'prediction');
    disp(nnz(pred(:)>1))
    %     for u = length(f):-1:1
    %         pred(pred==1+u) = f(u)+1;
    %     end
%     softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
    %   [~,pred] = max(scores,[],3) ;
    ok = lb > 0 ;
    %     if isempty(confusion)
    %         N = size(softmaxScores,3);
    %         confusion = zeros(N);
    %     end
    confusion = confusion + accumarray([lb(ok),pred(ok)],1,[N N]) ;
% % %     if (mod(ii,1)==0)
% % %         clf; imagesc2(bsxfun(@rdivide,confusion,sum(confusion,2)));
% % %         title(sprintf('%d / %d, %f', ii, length(test_set), sum(diag(confusion))/sum(confusion(:))));
% % %         drawnow
% % %     end

    %     dpc(.01)
%     s = col(softmaxScores(:,:,end));
    % % %     s = sort(s,'descend');
%     L = imdb.labels{imId}(:);
    % % %     q = length(s)/10;
    % % % %     s = s(1:q);
    % % % %     L = L(1:q);
    % % %
%     all_scores{end+1} = s;
%     all_masks{end+1} = L;
%     all_inds{end+1} = ii*ones(size(L));
    
    %     figure(1);clf; imagesc2(sc(cat(3,softmaxScores(:,:,end),rgb/255),'prob_jet'));
    %     figure(2); clf; imagesc2(sc(cat(3,pred>1,rgb/255),'prob_jet'));
    %     dpc
end

% confusion = [];
cm_normalized = bsxfun(@rdivide,confusion,sum(confusion,2));
all_scores = cat(1,all_scores{:});
all_masks = cat(1,all_masks{:});
%

% [prec,rec,info] = vl_pr(  2*(single(all_masks)==3)-1,all_scores);
%
% aps = info.ap;

