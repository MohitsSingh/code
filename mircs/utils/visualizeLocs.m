function [p,p_mask] = visualizeLocs(conf,ids,locs,height,inflateFactor,add_border)
% visualizeLocs
if (nargin < 6)
    add_border = true;
end

if (nargin < 5)
    inflateFactor = 1; % default is no inflation
end

if (nargin < 4)
    height = 64;
end

r = 0;

p = {};
for k = 1:size(locs,1)
    k
    imageID = ids{locs(k,11)};
    if (ischar(imageID))
        [I,xmin,ymin,xmax,ymax] = toImage(conf,getImagePath(conf,imageID),1);
    else
        I = imageID;
    end
    %     if (locs(k,conf.consts.FLIP))
    %         I = flip_image(I);
    %     end
    size_ratio = (ymax-ymin+1)/conf.max_image_size;
    rect = inflatebbox(locs(k,1:4),inflateFactor);
    if (size_ratio > 1) % loc has been taken from cropped image
        if (locs(k,conf.consts.FLIP))
            rect = flip_box(rect,[ymax-ymin+1,xmax-xmin+1]/size_ratio);
        end
        rect = rect*size_ratio;        
    end
    rect([1 3]) = rect([1 3]) + xmin;
    rect([2 4]) = rect([2 4]) + ymin;
    
    [I_cropped,I_mask] = cropper(I,round(rect));
    if (locs(k,conf.consts.FLIP))
        I_cropped = flip_image(I_cropped);
    end
    
    I_cropped = myResize(I_cropped,height);
    %     imshow(I_cropped);
    I_mask = imresize(I_mask,[height NaN],'nearest');
    if (add_border)
        if (~isempty(strfind(imageID,conf.classes{conf.class_subset})))
            I_cropped =addBorder(I_cropped,3,[0 1 0]);
        else
            I_cropped =addBorder(I_cropped,3,[1 0 0]);
        end
        I_cropped =addBorder(I_cropped,1,[0 0 0]);
    end
    
    p{k} = I_cropped;
    p_mask{k} = I_mask;
end


