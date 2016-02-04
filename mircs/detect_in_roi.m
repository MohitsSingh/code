function [responses]=detect_in_roi(conf,clusters,imageData,currentID,debug_)

if (nargin < 5)
    debug_ = false;
end
responses = struct('imageID',{},'scoremaps',{},'rois',{});
responses(1).imageID = currentID;
posemap = 90:-15:-90;

[M,landmarks,face_box] = getSubImage(conf,imageData,currentID);
if (isempty(M))
    return;
end
curPose = posemap(landmarks.c);
%MM = imresize(M,[180 NaN],'bilinear');
% MM = getImage(conf,currentID);
MM = imresize(M,[240 NaN],'bilinear');
% MM = imresize(M,[120 NaN],'bilinear');
conf.detection.params.detect_keep_threshold = -inf;
conf.detection.params.detect_min_scale = .1;
conf.detection.params.detect_max_scale = 1;
% conf.features.winsize = [5 5];
conf.detection.params.detect_add_flip = 1;

responses(1).M = M;
responses(1).scoremaps = {};
responses(1).rois = {};
curMax = 0;
thetaRange = -20:10:20;
% thetaRange = -10:10:10;
% thetaRange = -6;
responses(1).thetas = thetaRange;
conf.clustering.min_cluster_size = 0;


%warps = [.8 1 1.2];
G = fspecial('gauss',size2(MM),size(MM,1)/6);
G = G/max(G(:));

warps = 1;
for rot = thetaRange
    
    %     I = I(end/3:2*end/3,:,:);
    %     I = I(:,end/4:3*end/4,:);
    
    for iWarp = 1:length(warps)
        for jWarp = 1:length(warps)
            
            I = imrotate(MM,rot,'bilinear','crop');
            if (length(warps) > 1 && warps(iWarp) ~=1 && iWarp == jWarp) % dont need isotropic scaling warps
                continue;
            end
            %             box_orig = bb = [1 1 size(I,2) size(I,2)];
            
            H = eye(3);
            H(1,1) = warps(iWarp);
            H(2,2) = warps(jWarp);
            
            I = imtransform2(I,H,'show',0,'pad','replicate');
            
            
            bb = [1 1 size(I,2) size(I,2)];
            dd = .7;
            bb = round(clip_to_image(inflatebbox(bb,[dd dd],'both',false),I));
            
            
            %                 I = cropper(I,bb);
            %     I = I(ceil(end*(1-dd)):floor(dd*end),:,:);
            %     I = I(:,ceil(end*(1-dd)):floor(dd*end),:);
            
            I = max(0,min(1,I));
            
            qq = applyToSet(conf,clusters,{I},[],'tmp','override',false,'nDetsPerCluster',1,...
                'uniqueImages',false,'visualizeClusters',false,'toSave',false);
            zz = cat(1,qq.cluster_locs);
            
            
            bc = round(boxCenters(zz));
            g_weights = G(sub2ind2(size2(I),fliplr(bc)));
            
            
            % get the looking direction and use it as a region of interest...
            if (isempty(zz))
                H = zeros(size2(I));
            else
                zz = zz(:,[1:4 12]);
                zz(:,1:4) = clip_to_image(zz(:,1:4),I);
                zz(:,5) = zz(:,5).*g_weights;
                H = computeHeatMap(I,zz,'max');
            end
            
            
            responses.scoremaps{end+1} = H;
            if (debug_)
                clf;subplot(1,3,1); imagesc(I); axis image;
                subplot(1,3,2); imagesc(sc(cat(3,H,I),'prob')); axis image;
                
                title(['rot: ' num2str(rot), 'max score: ' num2str(max(H(:)))]);                                                
            end
            pp = curPose+rot-10;
            vec = [sind(pp),cosd(pp)];
            %roi = directionalROI(I,[size(I,2)/2,0.8*size(I,1)/4],vec',40);
            roi = directionalROI(I,[size(I,2)/2,.4*size(I,1)],vec',50);
            responses.rois{end+1} = roi;
%             
            roi_weights = roi(sub2ind2(size2(I),fliplr(bc)));
            zz(:,5) = zz(:,5).*roi_weights;
            H = computeHeatMap(I,zz,'max');
            
            H = H.*roi;
            if (debug_)
                subplot(1,3,3); imagesc(sc(cat(3,H/max(H(:)),I),'prob')); axis image;
                title(['clipped score: ' num2str(max(H(:)))]);
                m = max(H(:));
                if (curMax < m)
                    curMax = m;
                end
                
                %         end
                pause
            end
        end
    end
end