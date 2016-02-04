function [model,param] = learn_detect_obj_structured(I_sub_poss,my_mouth_corners,poly_centers,angles,param)
% output is x,y,theta
% model definition and initialization
patterns = {};
labels = {};
for t = 1:length(I_sub_poss)
    patterns{t} = struct('img',I_sub_poss{t},'mouth_center',mean(my_mouth_corners{t}));
    labels{t} = [poly_centers(t,:) angles(t)];
end

param.patterns = patterns ;
% no rotations for now!
param.labels = labels;
param.lossFn = @my_lossCB ;
param.constraintFn  = @my_constraintCB;
param.featureFn = @my_featureCB;
dummy_ = vl_hog(zeros([fliplr(param.windowSize) 3],'single'),param.cellSize);
param.w_shape = size(dummy_);
param.nHog = numel(dummy_);
if param.offset_ker_map
    param.dimension = param.nHog+15;
else
    param.dimension = param.nHog+6;
end
param.verbose = 0;
% param.model

%model = svm_struct_learn('-c .1 -o 2 -v 1 -w 0 ', param) ;
model = svm_struct_learn('-c 1 -o 2 -v 1', param) ;
w = model.w;
w_hog = w(1:param.nHog);
% w_offset = w(param.nHog+1:end);
w_hog = single(reshape(w_hog,param.w_shape));
w_img = vl_hog('render',w_hog);
x2(w_img)
% for u = 1:length(I_sub_poss)
%     x = patterns{u};
%     [detections,scores] = detect_with_offset(x,model,param);
%     [z,iz] = sort(scores,'descend');
%     z = visualizeTerm(scores,boxCenters(detections'),size2(x.img));
% end
