function [images,inds] = multiRead(conf,baseDir,ext,ids,sz,maxImages)
if (nargin < 6)
    maxImages = 1000;
    warning('maxImages set to 1000')
end
if (nargin < 5)
    sz = [];
end
if (nargin <4 || isempty(ids))
    
    if (nargin < 3 || isempty(ext))
        ext = '.jpg';
    end
    images = {};
    fileNames = dir(fullfile(baseDir,['*' ext]));
    for k = 1:min(maxImages,length(fileNames))
%         k
        I =imread(fullfile(baseDir,fileNames(k).name));
        if (~isempty(sz))
            if (isscalar(sz))
                sz = [sz sz];
            end
            I = imResample(I,sz,'bilinear');
        end
        images{k} = I;
        inds{k} = str2num(fileNames(k).name(1:end-4));
    end
else
    inds = [];
    images = {};
    for k = 1:length(ids)
        [~,t,~] = fileparts(ids{k});
        images{k} = imread(fullfile(baseDir,[t,ext]));
    end
end

% for k = 1:length(images)
%     I = images{k};
%     
%     %     I = im2double(gray2rgb(I));
%     
%     %     if size(I,1)/conf.max_image_size > 1
%     %         I = resize(I,conf.max_image_size/size(I,1));
%     %     end
%     
%     I = min(I,1);
%     I = max(I,0);
%     images{k} = I;
% end

    function I = gray2rgb(I)
        if (length(size(I))==2)
            I = repmat(I,[1 1 3]);
        end
    end


end