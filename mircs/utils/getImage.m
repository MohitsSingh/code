function [ I,xmin,xmax,ymin,ymax ] = getImage( conf,ID ,fullOverride,sizeOnly)
%GETIMAGE Summary of this function goes here
%   Detailed explanation goes here

if (iscell(ID) && length(ID)==1)
    ID = ID{1};
end

if (isnumeric(ID) || islogical(ID))
    I = ID;
else
    if (isstruct(ID))
        ID = ID.imageID;
    end
    xmin = 0; xmax = 0; ymin = 0; ymax=  0;
    if (nargin < 3 || isempty(fullOverride))
        if (isfield(conf,'get_full_image'))
            fullOverride = conf.get_full_image;
        else
            fullOverride = false;
        end
    end
    if (nargin < 4)
        sizeOnly = false;
    end
    %     check source of image
    %     if (exist(imagePath,'file'))
    %         I = toImage(conf,imagePath);
    imagePath = getImagePath(conf,ID);
    if (exist(imagePath,'file'))
        [I,xmin,ymin,xmax,ymax] = toImage(conf,imagePath,fullOverride,sizeOnly);
    else
        imagePathPascal = getImagePathPascal(conf,ID);
        if (exist(imagePathPascal,'file'))
            I = toImagePascal(conf,imagePath);
        else
            error(['couldn''t find image with image id ' ID ' anywhere!']);
        end
    end
end

if (nargout == 2) % return as rect
    xmin = [xmin ymin xmax ymax];
    
end

if (length(size(I))==2)
    I = cat(3,I,I,I);
end

end

