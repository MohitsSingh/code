
function [ I,xmin,ymin,xmax,ymax ] = toImagePascal(conf, I)%,get_full_image)
%TOIMAGE Summary of this function goes here
%   Detailed explanation goes here
get_full_image = strcmp(conf.pasMode,'test');
if (~ischar(I))
    xmin = 0;xmax = 0;ymin = 0;ymax= 0;
    return;
end

ischar_ = 0;
if (~exist('get_full_image','var'))
    get_full_image = 0;
end
if (ischar(I)) % read the image and crop out the action!
    ischar_ = 1;
    [~,fname,~] = fileparts(I);
    I=imread(getImagePathPascal(conf,fname));
    xmin = 1;
    ymin = 1;
    xmax = size(I,2);
    ymax = size(I,1);
    if (~get_full_image)
        if (isfield(conf,'pasClass'))
            rec = PASreadrecord(sprintf(conf.VOCopts.annopath,fname));
            t = strcmpi({rec.objects.class},conf.pasClass); %TODO - a hack
            t = find(t,1,'first');
            if (any(t))
                xmin = rec.objects(t).bndbox.xmin;
                ymin = rec.objects(t).bndbox.ymin;
                xmax = rec.objects(t).bndbox.xmax;
                ymax = rec.objects(t).bndbox.ymax;
                get_full_image = 0;
            end
        else
            get_full_image = 1;
        end
    end
    %     if(get_full_image)
%     xmin = 1;
%     ymin = 1;
%     xmax = size(I,2);
%     ymax = size(I,1);
    %     end
    
end
I = toI(I);
if (ischar_ && ~get_full_image)
    I = I(ymin:ymax,xmin:xmax,:);
end

% downsize if necessary...
if size(I,1)/conf.max_image_size > 1% && ~get_full_image
    I = resize(I,conf.max_image_size/size(I,1));
end

I = min(I,1);
I = max(I,0);
end