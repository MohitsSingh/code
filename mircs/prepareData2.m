function [phis,Is] = prepareData2(img_data)
% create a mapping from keypoints to indices.
kp_data = [img_data.kp_data];
pt_names = unique(cat(1,kp_data.pointNames));
pt_map = containers.Map(pt_names,1:length(pt_names));
nPts = length(pt_names);
nData = length(img_data);
phis = nan(nData,nPts,2);
boxes = cat(1,img_data.face_det);
boxes = boxes(:,1:4);
boxes = inflatebbox(boxes,1.3,'both',false);
boxes = round(boxes);
%IsTr = multiCrop(conf,{data_train.image_path},boxes);
Is = cell(nData,1);
ticID = ticStatus('preparing training data for landmark localization...',.5,.5);
for u = 1:nData
    curPts = img_data(u).kp_data;
    xy = curPts.pts;
    pt_indices = cellfun(@(x) pt_map(x),curPts.pointNames);
    phis(u,pt_indices,1) = xy(:,1)-boxes(u,1)+1;
    phis(u,pt_indices,2) = xy(:,2)-boxes(u,2)+1;
    I = imread(img_data(u).image_path);
    Is{u} = cropper(I,boxes(u,:));
    tocStatus(ticID,u/nData);
    %     clf; imagesc2(Is{u});
    %     showAnnotation(phis(u,:,:));
    %     pause;
end
% phis(:,:,1) = bsxfun(@minus,phis(:,:,1),;
% phis(:,:,2) = bsxfun(@minus,phis(:,:,2),boxes(:,2))+1;
end
