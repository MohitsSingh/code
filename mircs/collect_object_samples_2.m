function [initialSamples,all_boxes,all_ims,obj_ims] = collect_object_samples_2(conf,dataset,curClass)
conf.clustering.sample_method = 'none';
conf.clustering.min_hog_energy = 0;
images = dataset_list(dataset,'train',curClass);
all_x = {};
all_ims = {};
all_boxes = {};
image_paths = {};
obj_ims = {};
progress;
% images = vl_colsubset(images,100);
for k = 1:length(images)
    k
    progress(k,length(images))
    [bbox,im]=  dataset_image(dataset,curClass,images{k});
    %     clf; imagesc(im); axis image; pause; continue;
    bb2 = bbox; bb2(:,3:4) = bb2(:,3:4)+bb2(:,1:2);
    curObjIms = multiCrop(conf,{im},clip_to_image(inflatebbox(bb2,[1.5 1.5],'both',false),im));
    curObjIms = cellfun2(@(x) rescaleImage(x,180),curObjIms);
    obj_ims= [obj_ims;col(curObjIms)];
    
    % get only vertical aspect ratios
    %[boxes,X] = sampleBoxes(im,bbox);
    [boxes,X] = sampleBoxes_2(conf,im,bbox);
    if (isempty(boxes))
        continue;
    end
    all_ims = [all_ims;col(multiCrop(conf,{im},round(boxes(:,1:4))))];
    boxes(:,11) = k;
    all_boxes{k} = boxes;
    all_x = [all_x,{X}];
    size(boxes)
%         if (~isempty(boxes))
%             clf,imagesc(im); axis image; hold on; plotBoxes(boxes,'g');
%             pause;
%         end
end

%     fclose(fopen(sprintf('tmp_10/%03.0f',k),'w'));
% end

%save bottle_patches.mat all_ims all_x
initialSamples = cat(2,all_x{:});
% all_x = cat(2,all_x{:});
function [boxes,X] = sampleBoxes_2(conf,im,bbox)
% random boxes around vertical axis.
boxes = [];
X = [];
aspects = bbox(:,4)./bbox(:,3);
bbox = bbox(aspects > 1,:);
if (isempty(bbox))
    return;
end
% now get the top portion of the boxes only.
bbox(:,4) = bbox(:,4)/4;

bbox(:,3:4) = bbox(:,3:4)+bbox(:,1:2);

bc = boxCenters(bbox);
for k = 1:size(bc,1)
    r = min(bbox(k,3:4)-bbox(k,1:2));
    bbox(k,:) = inflatebbox(bbox(k,:),[r r],'both',true);
end

bbox = round(makeSquare(bbox));

bbox_c = clip_to_image(bbox,im);
bbox(sum(bbox_c~=bbox,2)>1,:) = [];
boxes = bbox;
sz = 8*conf.features.winsize;
% enlarge each box to accomodate the padding size...


tt = conf.features.winsize/(conf.features.winsize-conf.features.padSize);

boxes = inflatebbox(boxes,[tt tt],'both',false);
ims = multiCrop(conf,{im},boxes,sz);%cropper(im,boxes);
X = imageSetFeatures2(conf,ims,true,sz);

% for iBox = 1:size(bbox,1)
%     bc = boxCenters(bbox(iBox,:));
%     x_center = bc(1);
%
% end


function [boxes,X]=  sampleBoxes(im,bbox)
boxes = [];
X = [];


aspects = bbox(:,4)./bbox(:,3);
bbox = bbox(aspects > 1,:);
% now get the top portion of the boxes only.
bbox(:,4) = bbox(:,4)/4;
if (isempty(bbox))
    return;
end
bbox(:,3:4) = bbox(:,3:4)+bbox(:,1:2);
% inflate bboxes a bit and clip to image.
bbox = inflatebbox(bbox,[1+.2,1+.2/aspects],'both',false);
bbox = clip_to_image(bbox,im);

[im,scaleFactor] = rescaleImage(im,360); % make sure largest dimension is not too large.
% sample random features around the bbox.
bbox = bbox*scaleFactor;

% sample features with an overlap of at most .5 (per each scale)
conf.detection.params.detect_min_scale = .1;
[X,uus,vvs,scales,t,boxes ] = allFeatures( conf,im,1);
if (isempty(boxes))
    return;
end
intersection = BoxIntersection2(bbox,boxes);
[~,~,areas] = BoxSize(boxes);
areas = areas';
intersection = bsxfun(@rdivide,intersection,areas);

sel_ = max(intersection,[],1) > .8;

boxes = boxes(sel_,:);
X = X(:,sel_);

inds = randperm(size(boxes,1));
boxes = boxes(inds,:);
X = X(:,inds);    %
pick = nms(boxes,.5);
boxes=boxes(pick,:);
X = X(:,pick);


% for k = 1:length(ims)
%     clf; imagesc(ims{k}); axis image; pause;
% end

% % % older version
