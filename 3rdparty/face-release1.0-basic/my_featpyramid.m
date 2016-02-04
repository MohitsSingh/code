function [pyra,mask_pyra] = my_featpyramid(im, model, mask)
% pyra = featpyramid(im, model, padx, pady);
% Compute feature pyramid.
%
% pyra.feat{i} is the i-th level of the feature pyramid.
% pyra.scales{i} is the scaling factor used for the i-th level.
% pyra.feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

if (nargin < 3)
    mask = ones(size(im,1),size(im,2));
end
interval  = model.interval;
sbin = model.sbin;

% Select padding, allowing for one cell in model to be visible
% Even padding allows for consistent spatial relations across 2X scales
padx = max(model.maxsize(2)-1-1,0);
pady = max(model.maxsize(1)-1-1,0);
%padx = model.maxsize(2);
%pady = model.maxsize(1);
% padx = ceil(padx/2)*2;
% pady = ceil(pady/2)*2;

sc = 2 ^(1/interval);
imsize = [size(im, 1) size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/(5*sbin))/log(sc));
pyra.feat  = cell(max_scale + interval, 1);
mask_pyra.feat  = cell(max_scale + interval, 1);

pyra.scale = zeros(max_scale + interval, 1);
mask_pyra.scale = zeros(max_scale + interval, 1);

% our resize function wants floating point values
im = double(im);
for i = 1:interval
    scaled = resize(im, 1/sc^(i-1));
    %     mask_scaled = resize(mask, 1/sc^(i-1));
    % "first" 2x interval
    pyra.feat{i} = features(scaled, sbin/2);
    pyra.scale(i) = 2/sc^(i-1);
    sz = size(pyra.feat{i});sz = sz(1:2);
    mask_pyra.feat{i} = imresize(mask, sz(1:2),'nearest');
%     clf; imagesc(mask_pyra.feat{i}); axis image; pause;
    mask_pyra.scale(i) = 2/sc^(i-1);
    % "second" 2x interval
    pyra.feat{i+interval} = features(scaled, sbin);
    pyra.scale(i+interval) = 1/sc^(i-1);
    sz = size(pyra.feat{i+interval});sz = sz(1:2);
    mask_pyra.feat{i+interval} = imresize(mask, sz(1:2),'nearest');
    mask_pyra.scale(i+interval) = 1/sc^(i-1);
    % remaining interals
    for j = i+interval:interval:max_scale
        scaled = reduce(scaled);
        pyra.feat{j+interval} = features(scaled, sbin);
        sz = size(pyra.feat{j+interval});sz = sz(1:2);
        mask_pyra.feat{j+interval} = imresize(mask, sz(1:2),'nearest');
        pyra.scale(j+interval) = 0.5 * pyra.scale(j);
        mask_pyra.scale(j+interval) = 0.5 * mask_pyra.scale(j);
    end
end

for i = 1:length(pyra.feat)
    % add 1 to padding because feature generation deletes a 1-cell
    % wide border around the feature map
    pyra.feat{i} = padarray(pyra.feat{i}, [pady+1 padx+1 0], 0);
    % write boundary occlusion feature
    pyra.feat{i}(1:pady+1, :, end) = 1;
    pyra.feat{i}(end-pady:end, :, end) = 1;
    pyra.feat{i}(:, 1:padx+1, end) = 1;
    pyra.feat{i}(:, end-padx:end, end) = 1;
    % same for mask
    mask_pyra.feat{i} = padarray(mask_pyra.feat{i}, [pady+1 padx+1 0], 0);
%     mask_pyra.feat{i} = imdilate(mask_pyra.feat{i},ones(3));
    % write boundary occlusion features for mask
%     mask_pyra.feat{i}(1:pady+1, :, end) = 1;
%     mask_pyra.feat{i}(end-pady:end, :, end) = 1;
%     mask_pyra.feat{i}(:, 1:padx+1, end) = 1;
%     mask_pyra.feat{i}(:, end-padx:end, end) = 1;
end

pyra.scale    = model.sbin./pyra.scale;
pyra.interval = interval;
pyra.imy = imsize(1);
pyra.imx = imsize(2);
pyra.pady = pady;
pyra.padx = padx;

mask_pyra.scale    = model.sbin./pyra.scale;
mask_pyra.interval = interval;
mask_pyra.imy = imsize(1);
mask_pyra.imx = imsize(2);
mask_pyra.pady = pady;
mask_pyra.padx = padx;
