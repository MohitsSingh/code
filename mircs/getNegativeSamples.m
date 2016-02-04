function neg_feats = getNegativeSamples(conf,image_data_neg,obj_detector,param)

% extract the mouth regions and apply detection on negative image set
r = cat(1,image_data_neg.faceBox);
r = abs(param.poseMap(r(:,5))) <= 30;
image_data_neg_1 = image_data_neg(r);
[I_sub_negs,boxes_neg,factors_neg,mouth_pts_neg_local,mouth_pts_neg] = getSubImages(conf,image_data_neg_1,param);
% get detection candidates for these images (negative samples)
param.debugging = false;
get_neg_param = param;
get_neg_param.scales = 1;
neg_dets = run_detector(I_sub_negs,obj_detector,get_neg_param);
% myVisualize(conf,image_data_neg_1,mouth_pts_neg,[],[],[]);

% for z = 1:length(neg_dets)  % visualization of negative examples
%     figure(1),clf;imagesc2(I_sub_negs{z});
%     plotPolygons(neg_dets{z}(:,1:2),'r.');
% %     plotBoxes(negFaceBoxes(z,:));
%     plotPolygons(mouth_pts_neg_local{z},'g+');
%     pause
% end
neg_feats = {};
for r = 1:length(neg_dets)
    r
    cur_neg_dets = neg_dets{r};
    n = size(cur_neg_dets,1);
    neg_feats{r} = extract_straw_feats(repmat(mouth_pts_neg_local(r),n,1),...
        cur_neg_dets(:,1:2), cur_neg_dets(:,3));
    neg_feats{r} = [neg_feats{r}, cur_neg_dets(:,4)];
end

neg_feats = cat(1,neg_feats{:});function [ output_args ] = untitled2( input_args )

