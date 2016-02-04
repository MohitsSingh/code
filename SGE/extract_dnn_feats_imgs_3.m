function res = extract_dnn_feats_imgs_3(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd ~/code/mircs;
    %     initpath;
    %     config;
    addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
    addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox');
    addpath('~/code/common');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11/examples/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11/matlab/');
    addpath('/home/amirro/code/common');
    vl_setup;
    vl_setupnn;
    res.net = init_nn_network('imagenet-vgg-s.mat');
    for t = 1:length(res.net.layers)
        curLayer = res.net.layers{t};
        if (isfield(curLayer,'weights'))
            curLayer.filters = curLayer.weights{1};
            curLayer.biases = curLayer.weights{2};
            curLayer = rmfield(curLayer,'weights');
            res.net.layers{t} = curLayer;
        end
    end
    
    %     res.conf = conf;
    %     res.net_deep = init_nn_network('imagenet-vgg-verydeep-16.mat');
    
    return;
end

layers = 16;
net = initData.net;
% net = net(1:end-1); % we don't need the last layer...
I = im2double(imread(params.name));
nWindows = 5^2;
% wndSize = floor(size2(I)/sqrt(nWindows));
% j = round(max(1,wndSize/2));
feats_occ = {};
feats_windows = {};

rects = makeTiles(I,nWindows,2);
for iRect=1:size(rects,1)
    % for dx = 1:j(2):size(I,2)
    %     for dy = 1:j(1):size(I,1)
    %         iRect = iRect+1;
    curRect = rects(iRect,:);
    r = I;
    fprintf(1,'%d\n',iRect);
    %     curRect
    %     curRect = rects(iRect,:);
    %         r(yy(u):yy(u)+wndSize,xx(u):xx(u)+wndSize,:) = .5;
    r(curRect(2):curRect(4),curRect(1):curRect(3),:) = .5;
    ff = extractDNNFeats(r,net,layers);
    feats_occ{end+1} = ff.x;
    m = cropper(I,curRect);
    ff = extractDNNFeats(m,net,layers);
    feats_windows{end+1} = ff.x;
    %     z{iRect} = r;
    %     clf; imagesc2(r/255); dpc;
    %     end
end
feats_occ = cat(2,feats_occ{:});
feats_windows = cat(2,feats_windows{:});
ff = extractDNNFeats(I,net,layers);
res.feats_global = ff.x;
res.feats = feats_occ;
res.feats_windows = feats_windows;
res.rects = rects;
