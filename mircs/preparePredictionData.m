function [XX,offsets,all_scales,imgInds,subInds,values,kdtree,all_boxes,imgs] = preparePredictionData(conf,imgData,p)
conf.detection.params.detect_min_scale = 1;
conf.features.winsize = p.wSize;
scaleToPerson = p.scaleToPerson;
img_h = p.img_h; % we set the image height to a constant size to become somewhat scale invariant,
Xs = {};
offsets = {};
all_boxes = {}; % locations of boxes in each image
imgInds = {}; % index of image in current image set.
subInds = {}; % index of box in relevant image
imgs = {};
all_scales = {};
values = {};
for k = 1:length(imgData)
    [I,I_rect] = getImage(conf,imgData(k));           
    if (scaleToPerson)
        scaleFactor = img_h/(I_rect(4)-I_rect(2));
        
        if (p.normalizeWithFace)
            the_face = imgData(k).alternative_face;
            if (isempty(the_face))
                the_face=imgData(k).faceBox;
            end
            scaleFactor = .2*img_h/(the_face(4)-the_face(2));
        end
        
    else
        scaleFactor = img_h/size(I,1);
    end
    I = imResample(I,scaleFactor);
    imgs{k} = im2uint8(I);
    objBox = imgData(k).obj_bbox*scaleFactor;
    
    
%     [M,landmarks,face_box,face_poly,mouth_box,mouth_poly,xy_c] = ...
%     getSubImage(conf,imgData(k),1,false,true,false);
    
%     objBox = face_box*scaleFactor;
    
    [X,uus,vvs,scales,~,boxes ] = allFeatures( conf,I,1 );
    ovps = boxesOverlap(boxes,objBox);
    bc_obj = boxCenters(objBox);
    all_boxes{end+1} = boxes;
    Xs{end+1} = X;
    values{end+1} = ovps;
    subInds{end+1} = col(1:size(boxes,1));
    imgInds{end+1} = k*ones(size(X,2),1);
    offsets{end+1} = bsxfun(@minus,bc_obj,boxCenters(boxes));
    
    
    %% debugging
%         sel_ = 1:31:size(boxes,1);
%         curOffsets = offsets{end}(sel_,:);
%         clf;imagesc2(I); hold on; plotBoxes(boxes(sel_,:),'g--');
%         xy_start = boxCenters(boxes(sel_,:));
%         plotPolygons(xy_start,'r+');
%         quiver(xy_start(:,1),xy_start(:,2),curOffsets(:,1),curOffsets(:,2),0,'r');
%         pause;
    % %
    
    %%
    
    all_scales{end+1} = scales(:);
end

XX = cat(2,Xs{:});
offsets = cat(1,offsets{:});
all_scales = cat(1,all_scales{:});
imgInds = cat(1,imgInds{:});
subInds = cat(1,subInds{:});
values = cat(1,values{:});
kdtree = vl_kdtreebuild(XX,'Distance','L1');