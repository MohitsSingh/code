function trainSet = prepareForDPM(conf,true_ids,false_ids,pos_boxes,toFlip)
pos      = [];
impos    = [];
numpos   = 0;
numimpos = 0;
dataid   = 0;
conf.get_full_image = true;
tmpDir = '~/storage/tempImages';
if (nargin < 5)
    toFlip = false(size(true_ids));
end
for k = 1:length(true_ids)
    100*k/length(true_ids)
    numpos = numpos + 1;
    dataid = dataid + 1;
    if (ischar(true_ids{k}))
        pos(numpos).im = getImagePath(conf,true_ids{k});
    else
        pos(numpos).im = true_ids{k};
    end
    %[im,~,~,~,~] = getImage(conf,true_ids{k});    
    if (isempty(pos_boxes)) % then take the whole images, minus 25% from each side
       bbox = [1 1 dsize(pos(numpos).im,[2 1])];
       bbox = [inflatebbox(bbox,[.6 .6],'both',false) 0];
    else
        bbox = pos_boxes(k,:);
    end
          
    if (size(bbox,2) >= 5 && bbox(5) ~= 0)        
        I = imread(pos(numpos).im);
        z = false(dsize(I,1:2));
        bbox(1:4) = round(bbox(1:4));
        z(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;
        z = imrotate(z,bbox(5),'nearest','loose');
        [yy,xx] = find(z);        
        I = imrotate(I,bbox(5),'bicubic','loose');
        bbox = pts2Box([xx yy]);
        
        [~,name,ext] = fileparts(true_ids{k});
        newPath = fullfile(tmpDir,sprintf('%s%04.0f%s',name,numpos,ext));
        pos(numpos).im = newPath;
%      
%         displayRegions(im2double(I),{z},[],-1);
%         hold on; plotBoxes2(bbox([2 1 4 3]),'g');
%         pause
        imwrite(I,newPath);
    end
    
    if (toFlip(k))
        I = imread(pos(numpos).im);
        bbox = flip_box(bbox,size2(I));
    end
    if (ischar(true_ids{k}))    
        imInfo = imfinfo(pos(numpos).im);
    else
        imInfo.Width = size(pos(numpos).im,2);
        imInfo.Height = size(pos(numpos).im,1);
    end
    % create temp oriented image if necessary. 
    
    pos(numpos).x1 = bbox(1);
    pos(numpos).y1 = bbox(2);
    pos(numpos).x2 = bbox(3);
    pos(numpos).y2 = bbox(4);
    pos(numpos).boxes   = bbox;
    pos(numpos).flip    = toFlip(k);
    pos(numpos).trunc   = false;
    pos(numpos).dataids = dataid;
    pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    
    % Create flipped example
%     imgsize = size(im);
    numpos  = numpos + 1;
    dataid  = dataid + 1;
    oldx1   = bbox(1);
    oldx2   = bbox(3);
    bbox(1) = imInfo.Width - oldx2 + 1;
    bbox(3) = imInfo.Width - oldx1 + 1;
    
    pos(numpos).im      = pos(numpos-1).im;
    pos(numpos).x1      = bbox(1);
    pos(numpos).y1      = bbox(2);
    pos(numpos).x2      = bbox(3);
    pos(numpos).y2      = bbox(4);
    pos(numpos).boxes   = bbox;
    pos(numpos).flip    = ~toFlip(k);
    pos(numpos).trunc   = false;
    pos(numpos).dataids = dataid;
    pos(numpos).sizes   = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    
    dataid = dataid + 1;
    numimpos                = numimpos + 1;
    impos(numimpos).im      = pos(numpos).im;
    impos(numimpos).boxes   = pos(end-1).boxes;
    impos(numimpos).dataids = dataid;
    impos(numimpos).sizes = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    impos(numimpos).flip    = false;
    impos(numimpos).dataids = dataid;
    
    dataid = dataid + 1;
    numimpos                = numimpos + 1;
    impos(numimpos).im      = pos(numpos).im;
    impos(numimpos).boxes   = pos(end).boxes;
    impos(numimpos).dataids = dataid;
    impos(numimpos).sizes = (bbox(3)-bbox(1)+1)*(bbox(4)-bbox(2)+1);
    impos(numimpos).flip    = false;
    impos(numimpos).dataids = dataid;
    
    unflipped_boxes         = impos(numimpos-1).boxes;
end

neg = [];
numneg = 0;
for k = 1:length(false_ids)
    dataid             = dataid + 1;
    numneg             = numneg+1;
    if (ischar(false_ids{k}))
        neg(numneg).im     = getImagePath(conf,false_ids{k});
    else
        neg(numneg).im     = false_ids{k};
    end
    neg(numneg).flip   = false;
    neg(numneg).dataid = dataid;
end
trainSet.pos = pos;
trainSet.neg = neg;
trainSet.impos  = impos;

