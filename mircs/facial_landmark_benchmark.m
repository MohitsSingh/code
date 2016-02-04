%% comparison of my facial landmarks on fra db vs zhu/ramanan
addpath('/home/amirro/code/3rdparty/ojwoodford-export_fig-165dc92');
% initpath;
% config;
% load s40_fra;
% fra_db = s40_fra;
%%
load fra_db
roiParams.infScale = 1.5;
roiParams.absScale = 200;
roiParams.centerOnMouth = false;
%results = struct('zhu_bad',{},'ours_bad',{},'face_detected',{},'ind_ind_fra_db',{})
clear results;
% zhu_bad = true(size(fra_db));
% ours_bad = true(size(fra_db));
% face_detected = false(size(fra_db));
load s40_fra; fra_db = s40_fra;
frontal.zhu_nose_center = 6;
frontal.zhu_left_mouth = 35;
frontal.zhu_right_mouth = 44;
frontal.zhu_chin_center = 52;
frontal.zhu_left_eye = 12;
frontal.zhu_right_eye = 22;
frontal.zhu = [6 35 44 53 12 22];
side.zhu_nose_center = 3;
side.zhu_left_mouth = 18;
side.zhu_right_mouth = 18;
side.zhu_chin_center = 30;
side.zhu_left_eye = 10;
side.zhu = [3 18 18 30 10];
D = defaultPipelineParams(false);
toShow = true;

%%
results = struct;
u = 0;
toShow = true;
ppp = randperm(length(fra_db))
for t_p = 1:length(fra_db)
    t = ppp(t_p);
    curImgData = fra_db(t);
%         if (curImgData.classID~=conf.class_enum.DRINKING),continue,end
    ind_in_fra_db = curImgData.indInFraDB;
    if (ind_in_fra_db==-1),continue,end
    t
    u = u+1;
    %     if (curImgData.classID~=conf.class_enum.BRUSHING_TEETH),continue,end
    
    face_det = curImgData.raw_faceDetections.boxes(1,1:4);
    faceBox = curImgData.faceBox;
    faceBox_gt = curImgData.faceBox_gt;
    
    [ovps,ints] = boxesOverlap(faceBox,faceBox_gt);
    results(u).ovp = ovps;
    results(u).ind_int_fra_db = ind_in_fra_db;
    if (ovps > .3)
        results(u).face_detected = true;
    else
        results(u).face_detected = false;
        %         I_orig = getImage(conf,curImgData);
        %         clf;imagesc2(I_orig);
        %         plotBoxes(curImgData.faceBox,'g-','LineWidth',2);
        %         plotBoxes(face_det,'r-','LineWidth',2);
        %         pause;
%         I_orig = getImage(conf,curImgData);
%         clf; imagesc2(I_orig);
%         plotBoxes(faceBox,'r-');
%         plotBoxes(curImgData.faceBox_gt,'g-');pause
        continue
%         
    end
    
    
    %     pause;continue
    curImgData = switchToGroundTruth(curImgData);
    [I_orig,I_rect] = getImage(conf,curImgData);
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImgData,roiParams);
    face_det = face_det+I_rect([1 2 1 2]);    
    %     x2(I_orig);plotBoxes(curImgData.raw_faceDetections.boxes(:,:));
    %     x2(I_orig);plotBoxes(curImgData.faceBox);
    % load zhu,ramanan results
    R1 = j2m('~/storage/s40_my_facial_landmarks',curImgData);
    %     if (~exist(R1,'file'))
    %         continue
    %     end            
    %
    L = load(j2m('~/storage/fra_landmarks',curImgData.imageID));
    if (isfield(L,'res'))
        res = L.res;
    else
        res = L;
    end
    results(u).zhu_bad = true;
    TYPE_FRONTAL = 1;
    TYPE_PROFILE = 2;
    zhu_type = TYPE_FRONTAL;
    if (any(res.landmarks(2).s))
        gotZhu = true;
        zhu_polys = cellfun2(@mean,res.landmarks(2).polys);
        zhu_polys = cat(1,zhu_polys{:});
        if (length(zhu_polys)==68) % frontal
            %             plotPolygons(polys(frontal.zhu,:),'gd');
        elseif (length(zhu_polys)==39) % frontal
            %             plotPolygons(polys(side.zhu,:),'gd');
            zhu_type = TYPE_PROFILE;
        else
            error('unexpected number of points in zhu');
            break
        end
        %         showCoords(polys)
    else
        continue;
    end
    results(u).zhu_bad = false;
    results(u).zhu_type = zhu_type;
    [kp_preds,kp_pred_goods] = loadDetectedLandmarks(conf,curImgData);
    kp_centers = transformToLocalImageCoordinates(boxCenters(kp_preds(:,1:4)),scaleFactor,roiBox);
    
    results(u).zhu_res = zhu_polys;
    results(u).our_res = kp_centers([1 2 4 5 6 7],:);
        
    % transform ramanan keypoints to required keypoints
    if (zhu_type == TYPE_FRONTAL)
        zhu_ind = [frontal.zhu_left_eye,frontal.zhu_right_eye,frontal.zhu_left_mouth,...
            frontal.zhu_right_mouth,frontal.zhu_chin_center,frontal.zhu_nose_center];
    else
        zhu_ind = [side.zhu_left_eye,side.zhu_left_eye,side.zhu_left_mouth,...
            side.zhu_right_mouth,side.zhu_chin_center,side.zhu_nose_center];
    end
    zhu_pts = zhu_polys(zhu_ind,:);
    results(u).zhu_res = zhu_pts;
    %
    [kp_gt,goods] = loadKeypointsGroundTruth(curImgData,D.requiredKeypoints);
    kp_gt = kp_gt([1 2 4 5 6 7],:);
    results(u).gt_care = goods([1 2 4 5 6 7]);
    %     global_pred = transformToLocalImageCoordinates(global_pred(:,1:4),scaleFactor,roiBox);
    %     kp_global_centers = boxCenters(global_pred);
    %[rois1,roiBox1,I1,scaleFactor1,roiParams1] = get_rois_fra(conf,curImgData,roiParams);
    %     if (none(inBox(roiBox,kp_centers)))
    %         ours_bad = true;
    %         continue
    %     end
    
    our_pts = results(u).our_res;
    %     zhu_pts = results(u).our_res;
    gt_pts = transformToLocalImageCoordinates(kp_gt(:,1:2),scaleFactor,roiBox);
%     results(u).gt_care = kp_gt(:,3);
    results(u).gt_pts = gt_pts;
    results(u).delta_zhu = (gt_pts-zhu_pts);
    results(u).delta_ours = (gt_pts-results(u).our_res);
    results(u).zhu_error = mean(sum(results(u).delta_zhu(results(u).gt_care,:).^2,2).^.5);
    results(u).our_error = mean(sum(results(u).delta_ours(results(u).gt_care ,:).^2,2).^.5);
    results(u).curFaceSize =  norm(gt_pts(1,:)-gt_pts(5,:));           
    
    curFaceSize = results(u).curFaceSize;
    our_error = results(u).our_error;
    zhu_error = results(u).zhu_error;
    
    if (toShow)
        clf;
        vl_tightsubplot(1,2,1);imagesc2(I);
%         title(num2str(zhu_error))
        axis off
        plotPolygons(zhu_pts,'g.','MarkerSize',20);
%         plotPolygons(gt_pts,'rs','LineWidth',2);        
%         quiver(gt_pts(:,1),gt_pts(:,2),-results(u).delta_zhu(:,1),-results(u).delta_zhu(:,2),0,'g');        
        vl_tightsubplot(1,2,2);imagesc2(I);        
%         title(num2str(our_error))
        plotPolygons(results(u).our_res,'g.','MarkerSize',20);
        axis off
        im = export_fig;
%         plotPolygons(gt_pts,'rs','LineWidth',2);               
%         plotPolygons(gt_pts(1,:),'yd','LineWidth',3);
%         plotPolygons(gt_pts(5,:),'yd','LineWidth',3);
%         plotPolygons(gt_pts(results(u).gt_care,:),'c*');                
%         quiver(gt_pts(:,1),gt_pts(:,2),-results(u).delta_ours(:,1),-results(u).delta_ours(:,2),0,'g');       
        disp(curFaceSize)
        fprintf('curFaceSize: %f\nrel. zhu error: %f\nrel. our error: %f\n',...
            curFaceSize,zhu_error/curFaceSize,our_error/curFaceSize);        
%             outputImage=ScreenCapture();        
        %     plotPolygons(kp_global_centers,'md');
%         pause;
        imwrite(im,j2m('/home/amirro/notes/images/cvpr_2015/landmarks',curImgData,'.png'));
    end        
end

%%save landmark_bench.mat results

