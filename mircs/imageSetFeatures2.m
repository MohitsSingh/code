function [ res , sizes] = imageSetFeatures2( conf,images,flat,sz,verbose)
%IMAGESETFEATURES Summary of this function goes here
%   Detailed explanation goes here
res = {};
sizes = {};
if (nargin < 5)
    verbose = false;
end
% res = zeros(620,length(images),'single');
for ii = 1:length(images)
    if (verbose)
        disp(['calculating descriptors for first set: %' num2str(100*ii/length(images))]);
    end
    im = im2single(getImage(conf,images{ii}));
    if (~isempty(sz))
        if (sz(1)<0)
            im = imresize(flip_image(im),-sz,'bilinear');
        else
            im = imresize(im,sz,'bilinear');
        end
    end
    %X = double(vl_hog(im,conf.features.vlfeat.cellsize,'NumOrientations',9));
    sbin = conf.detection.params.init_params.sbin;
    X = double(fhog(im,conf.features.vlfeat.cellsize));
    
%     X = conf.detection.params.init_params.features(im,sbin);
    
    sizes{ii} = size(X);
    res{ii} = (X(:));
    %     res(:,ii) = (X(:));
end

if (nargin >= 3 && flat)
    res = cat(2,res{:});
end
end