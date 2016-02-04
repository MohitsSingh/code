function res = multiCrop(conf,images,rect,sz)
res = {};
if (~iscell(images))
    images = {images};
end
if (isempty(rect))
    sz_ = size(images{1});    
    rect = [1 1 sz_([2 1])];
end
if (size(rect,1)==1)
    rect = repmat(rect,length(images),1);
end
mySz = [];

if (nargin == 4)    
    mySz = sz;
end

if (isscalar(mySz))
            mySz = [mySz mySz];
end

nIters = length(images);
if (length(images) < size(rect,1)) % more than one patch from each image
    nIters = size(rect,1);
    imageInds = ones(1,nIters);   
else    
    imageInds = 1:nIters;
end

if (size(rect,2)>=11)
    imageInds = rect(:,11);
    nIters = length(imageInds);
end
rect(:,1:4) = round(rect(:,1:4));
for k = 1:nIters
%     k

    if (isempty(conf))
        im = images{imageInds(k)};
    else
        im = getImage(conf,images{imageInds(k)});
    end
%     clf;
%     subplot(1,2,1);imagesc(im);hold on; plotBoxes2(rect(k,[2 1 4 3]));    
    rotation = 0;
    if (size(rect,2)==13)
        rotation = rect(k,13);
    end
    
    
%     if (rotation~=0)
    %%im = imrotate(cropper(imrotate(im,rotation,'bilinear','loose'), rect(k,:)),-rotation,'bilinear');
    
    if rotation~=0
        im = cropper(im,rect(k,:));
    else
        im = imrotate(cropper(imrotate(im,rotation,'bilinear','loose'), rect(k,:)),-rotation,'bilinear');
    end
    
%     end
%     subplot(1,2,2);imagesc(im);
%     pause;
    if (~isempty(mySz))
        %im = imresize(im,mySz);        
        
        im = imResample(im,mySz);
    end
    res{k} = im2uint8(im);
end