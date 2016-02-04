
function [ I,xmin,ymin,xmax,ymax ] = toImage(conf, I,get_full_image,sizeOnly)
%TOIMAGE Summary of this function goes here
%   Detailed explanation goes here
% xmin = 1;xmax = [];ymin = 1;ymax= [];
if (~ischar(I))
    I = im2double(I);
    xmin = 0;xmax = 0;ymin = 0;ymax= 0;   
else
    if (nargin < 3)
        get_full_image = conf.get_full_image;
    end
    if (nargin < 4)
        sizeOnly = false;
    end
    
    
    ischar_ = 0;
    
    % if (~exist('get_full_image','var'))
    %     get_full_image == 0 || conf.get_full_image;
    % end
    isPascal = false;
    if (ischar(I)) % read the image and crop out the action!
        ischar_ = 1;
        [~,fname,~] = fileparts(I);
        xml_path = fullfile(conf.xmlDir,[fname '.xml']);
        if (~exist(xml_path,'file'))
            isPascal = true;
            I = im2double(imread(I));
            ischar_ = 0;
            xmin = 0;ymin = 0;
            xmax = size(I,2);
            ymax = size(I,1);
            %         [~,I,~] = fileparts(I);
            
            %         rec = PASreadrecord(sprintf(conf.VOCopts.annopath,I));
             
        else
            %         if (~conf.not_crop)
            a = loadXML(xml_path);
            xmin = str2num(a.annotation.object.bndbox.xmin);
            xmax = str2num(a.annotation.object.bndbox.xmax);
            ymin = str2num(a.annotation.object.bndbox.ymin);
            ymax = str2num(a.annotation.object.bndbox.ymax);
            %         end
        end
        
        if (sizeOnly)
            I = [];
            return;
        end
    end
    if (ischar_)
        I = im2double(imread(I));
    end
%     I = toI(I);
    if (ischar_ && ~get_full_image)
        I = I(ymin:ymax,xmin:xmax,:);
    end
    
    % downsize if necessary...
end
if size(I,1)/conf.max_image_size > 1
    I = imResample(I,conf.max_image_size/size(I,1),'bilinear');
end

I = min(I,1);
I = max(I,0);

end