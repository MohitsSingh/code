function [phis,Is,boxes] = prepareData(phis,Is,boxes)
% converts bounding boxes to my format, crops out face images and
% transforms face points to bounding box coordinate system.


phis = reshape(phis,size(phis,1),[],3);
boxes(:,3:4) = boxes(:,3:4)+boxes(:,1:2);
% for u = 1:length(Is)
%     %     u   
%     clf; imagesc2(Is{u}); colormap gray;
%     showAnnotation(phis(u,:,:),boxes(u,:));
%     pause;
% end

boxes_before = boxes;
boxes_square = makeSquare(boxes);

%top_lefts = zeros(size(boxes,1),2);
% move each box so the center of the box is on the center of 
bounding_boxes = zeros(size(boxes));
for t = 1:size(bounding_boxes)
    bounding_boxes(t,:) = pts2Box(squeeze(phis(t,:,1:2)));
end
centers = boxCenters(bounding_boxes);%squeeze(mean(phis(:,:,1:2),2));
r = boxCenters(boxes_square);
boxes_square = boxes_square+repmat(centers-r,1,2);
% min(phis(:,:,1:2),2)
boxes = inflatebbox(boxes_square,1.4,'both',false); % was 1.2
% for u = 1:length(Is)
%     clf; imagesc2(Is{u}); colormap gray;
%     showAnnotation(phis(u,:,:),boxes(u,:));        
%     pause;
% end

boxes = round(boxes);
phis(:,:,1) = bsxfun(@minus,phis(:,:,1),boxes(:,1))+1;
phis(:,:,2) = bsxfun(@minus,phis(:,:,2),boxes(:,2))+1;

% re-center each box to be around the phi's

% 
for u = 1:length(Is)
    %     u
    Is{u} = cropper(Is{u},boxes(u,:));
%     clf; imagesc2(Is{u}); 
%     showAnnotation(phis(u,:,:));
%     pause;
end
