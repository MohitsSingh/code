function [I_subs,boxes,factors,all_mouth_pts_crop,all_mouth_pts] = getSubImages(conf,image_data,param)
% scales = 1;
debugging = true;
inflate_factor = param.inflate_factor;
I_subs = {};
all_mouth_pts_crop = {};
all_mouth_pts = {};
mouth_corner_inds = [35 41];
boxes = zeros(length(image_data),4);
factors = zeros(length(image_data),1);
scales = param.scales;
for t = 1:length(image_data)    
    %     cur_poly = image_data(t).gt_obj;
    I_orig = getImage(conf,image_data(t));
    faceBox = image_data(t).faceBox;
    xy_global = image_data(t).landmarks;
    curPose = param.poseMap(faceBox(5));
    mouth_pts = xy_global(mouth_corner_inds,:);
    all_mouth_pts{t} = mouth_pts;
    crop_box = round(image_data(t).faceBox(1:4));
    crop_box = inflatebbox(crop_box,inflate_factor,'both',false);
    crop_box = round(crop_box);
    boxes(t,:) = crop_box;
    I_sub = cropper(I_orig,crop_box);
    curResizeFactor = param.out_img_size(1)/size(I_sub,1);
    factors(t) = curResizeFactor;
    I_sub = im2single(imResample(I_sub,curResizeFactor,'bilinear'));    
    xy_crop = curResizeFactor*bsxfun(@minus,image_data(t).landmarks,crop_box(1:2));
    mouth_pts_crop = xy_crop(mouth_corner_inds,:);
    
    I_subs{t} = I_sub;
    all_mouth_pts_crop{t} = mouth_pts_crop;
    
    
    toShow = 0;
    if (toShow)
        clf; imagesc2(I_sub);
        plotPolygons(mouth_pts_crop,'r+');
        drawnow; 
    end
    continue
%     dpc;
    
    sc_total = [];
    [scales,iscales] = sort(scales,'ascend');
    %     scales = 1;
    figure(1);
    all_dets = {};
    for r = param.rotations%-60:10:60
        %         r
        II = imrotate(I_sub,r,'bilinear','crop');
        for iScale = 1:length(scales)
            curScale = scales(iScale);
            [detections, scores, hog, f] = detect(II, patchModels, param.cellSize, curScale);
            f = imrotate(f,-r,'bilinear','crop');
            if (isempty(sc_total))
                sc_total = -inf(size(f));
                %                 sc_total = -inf(size2(I_sub));
                rad = size(sc_total,1)/2;
                [xx,yy] = meshgrid(1:size(sc_total,1),1:size(sc_total,2));
                z = ((xx-rad).^2+(yy-rad).^2).^.5 <rad;
            end
            f = imResample(f,size(sc_total),'bicubic');
            sc_total = max(f,sc_total);
            all_dets{end+1} = [detections;r*ones(size(scores));scores]';
        end
        %         clf;vl_tightsubplot(1,2,1);imagesc2(II)
        %         vl_tightsubplot(1,2,2); imagesc2(sc_total);
        %         drawnow;
        %                             pause(.01);
        continue
    end
    all_dets = cat(1,all_dets{:});
    sc_total(~z) = -inf;
    vl_tightsubplot(1,2,1);imagesc2(I_sub)
    plotPolygons(xy_crop,'r.');
    plotPolygons(mouth_pts_crop,'g+');
    vl_tightsubplot(1,2,2); imagesc2(sc_total);
    dpc;
end
