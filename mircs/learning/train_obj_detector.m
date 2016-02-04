function [w,pos_images,pos_scores] = train_obj_detector(conf,pos_images_1,image_data_neg,landmarks_gt_path,obj_gt_baseDir,param)

[pos_images,polys,angles] = get_pos_samples(conf,pos_images_1,param,obj_gt_baseDir);
% now we have for each face the position of mouth corners and the x,y,theta
% of the straw : normalize w.r.t mouth coordinates
%[I_sub_poss,pos_boxes,pos_factors,~] = getSubImages(conf,pos_images_1,param);
%%
param.debugging = false;
neg_faces = [{image_data_neg.I}];
% neg_1_faces = multiResize(neg_1_faces,param.out_img_size);

pos_image_hogs = cellfun2(@(x) col(vl_hog(x, param.cellSize)), pos_images);
pos_image_hogs = cat(2,pos_image_hogs{:});
if param.cross_val~=0
    indices = crossvalind('kfold',length(pos_images),param.cross_val);
    pos_scores = zeros(size(pos_images_1));
    for t = 1:param.cross_val
        test_sel = indices==t;
        train_sel = indices~=t;
        w = train_detector(pos_images(train_sel),neg_faces,param);
        pos_scores(test_sel) = w(:)'*pos_image_hogs(:,test_sel);
    end
    % finally, train using all examples.
    w = train_detector(pos_images,neg_faces,param);
else%,varargin)
    w = train_detector(pos_images,neg_faces,param);
    pos_scores = [];
end
% imagesc2(vl_hog('render',w));colormap gray
end

