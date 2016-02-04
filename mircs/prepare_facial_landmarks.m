function imgData = prepare_facial_landmarks(conf,face_dets,imgData,facialLandmarkData,param)
for iImg =  1:1:length(imgData)
    for u = 1
        iImg                
        [I_orig,I_rect] = getImage(conf,imgData(iImg));        
        [xy_global,xy_local,faceBox,curPose, I, resizeFactor] = extract_facial_landmarks(I_orig,face_dets(iImg),facialLandmarkData);
        imgData(iImg).faceBox = faceBox;
        imgData(iImg).landmarks = xy_global;
        imgData(iImg).I = I;
        imgData(iImg).landmarks_local = xy_local;
        imgData(iImg).pose = curPose;
        if abs(curPose)>=30
            disp('skipping profile faces for now...');
            continue
        end
        
        [mouth_img,mouth_rect,mouth_pts,xy_near_mouth] = get_mouth_img(I_orig,xy_global,curPose,I,resizeFactor,param.imgSize);
        
        mmm = 2;
        nnn = 3;
        toVisualize = false;
        if (toVisualize)
            figure(1);clf
            vl_tightsubplot(mmm,nnn,1);imagesc2(clip_to_bounds(I));
            vl_tightsubplot(mmm,nnn,2);imagesc2(clip_to_bounds(I));
            plotPolygons(xy_local,'gd','LineWidth',2);
            vl_tightsubplot(mmm,nnn,3);
            imagesc2(I_orig);
            plotPolygons(xy_global,'g.','LineWidth',2);
            plotPolygons(mouth_pts,'r+','LineWidth',2);
            %             plotPolygons(gt_poly,'r-');
            plotBoxes(mouth_rect);
            vl_tightsubplot(mmm,nnn,4);
            imagesc2(mouth_img);
            plotPolygons(xy_near_mouth,'g.');
            %             plotPolygons(xy_near_mouth(mouth_corner_inds,:),'r+','LineWidth',2);
            m = rgb2gray(mouth_img);
            vl_tightsubplot(mmm,nnn,5);
            imagesc2(edge(m,'canny'));
            pause
        end
    end
end
