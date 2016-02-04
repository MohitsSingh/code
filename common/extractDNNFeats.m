function [res] = extractDNNFeats(imgs,net,layers,prepareSimple, mute,useGPU,rotation)
if nargin < 7
    rotation = 0;
end
if nargin < 6
    useGPU = false;
end
if nargin < 5
    mute = false;
end
if (nargin < 4)
    prepareSimple = false;
end
if nargin < 3
    layers = [16 19];
end

if (~iscell(imgs))
    imgs = {imgs};
end


% check the range of the first image to make sure it's uint8, 0-255 or
% single
img = imgs{1};
if ~isstr(img) && ~isa(img,'uint8')
    if max(img(:))>1
        error('expected either uint8 images or single/double images with range [0...1]');
    end
end

% split into batches of 256...
res = struct('layer_num',{},'x',{});
batchSize = 64;
if max(layers) > 25
    batchSize = 16;
end
if (~mute)
    %disp(['n images: ' num2str(length(imgs))]);
    tic_id = ticStatus('extracting deep features',.2,.1);
end
x_layers = cell(size(layers));
batches = batchify(length(imgs),ceil(length(imgs)/batchSize));
warning('rotating by -90 degrees!!!!');
for iBatch = 1:length(batches)
    batch = imgs(batches{iBatch});
    
    
    if rotation~=0
        batch = cellfun2(@(x) imrotate(x,rotation,'nearest'),batch);
    end
    
    % for t=1:batchSize:length(imgs)
    %         t/length(imgs)
    %     batch = imgs(t:min(t+batchSize-1, length(imgs)));
    
    for u = 1:length(batch)
        if (ischar(batch{u}))
            batch{u} = imread(batch{u});
        end
    end
    %profile on
    %     tic
    imo = prepareForDNN(batch,net,prepareSimple);
    %profile viewer
    %     toc
    if useGPU
        imo = gpuArray(imo);
    end
    %     tic
    dnn_res = vl_simplenn(net, imo);
    %     toc
    for iLayer = 1:length(layers)
        r = gather(dnn_res(layers(iLayer)).x);
        x_layers{iLayer}{end+1} = reshape(squeeze(r),[],length(batch));
        %         reshape(dnn_res(layers(iLayer)).x,[],batchSize);
    end
    if (~mute)
        tocStatus(tic_id,iBatch/length(batches));
    end
end
% fprintf('\n');

for iLayer = 1:length(layers)
    u = x_layers{iLayer};
    res(iLayer).x = cat(2,u{:});
    res(iLayer).name = net.layers{iLayer+1}.name;
    res(iLayer).layer_num = layers(iLayer);
end
