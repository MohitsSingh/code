function [centers,scores,angles] = detect_with_offset(x,model,param)
% add to each detection the location score...
w = model.w;
w_hog = w(1:param.nHog);
w_offset = w(param.nHog+1:end);
if (~param.use_mouth_offset)
    w_offset = zeros(size(w_offset));
end
I = x.img;
w_hog = single(reshape(w_hog,param.w_shape));
%param.avoid_out_of_image_dets = true;
detections = double(run_detector_nonms(I,w_hog,param));
centers = detections(:,1:2);
angles = detections(:,3);
scores = detections(:,4);

% mm = repmat(x.mouth_center,size(centers,1),1);
mm = repmat(x.mouth_center,size(centers,1),1)/param.offset_factor;
yy = centers/param.offset_factor;
s2 = 1.4142135624;
mouth_offset = mm-yy;
mouth_offsets = [mouth_offset.^2 s2*mouth_offset(:,1).*mouth_offset(:,2) s2*mouth_offset(:,1) s2*mouth_offset(:,2) ones(size(yy(:,1)))];
% mouth_offset = (y(1:2)-mm)/100;
%mouth_offset = [mouth_offset mouth_offset.^2 1];
% mouth_offsets = [yy, yy.^2, -2*mm.*yy, mm.^2];

% mouth_offset = bsxfun(@minus,centers,mm)/100;%/size(I,1);
% %mouth_offsets = [mouth_offsets mouth_offsets.^2 ones(size(mouth_offsets,1),1)];
%mouth_offsets = [mouth_offsets mouth_offsets.^2 ones(size(mouth_offsets,1),1)];
% mouth_offset = [mouth_offset, mouth_offset.^2, -2*(mm.*centers(1:2))/100^2, (mm/100).^2];
% mouth_offset = [mouth_offset, mouth_offset.^2, -2*(mouth_offset.*y(1:2))/100^2, (x.mouth_center/100).^2];


if (param.offset_ker_map)
    mouth_offsets = vl_homkermap(mouth_offsets',1)';
end
offset_scores = mouth_offsets*w_offset;

scores = scores+offset_scores;

