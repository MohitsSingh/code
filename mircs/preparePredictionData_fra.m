function [XX,FF,offsets,all_scales,imgInds,subInds,values,kdtree,all_boxes,imgs,masks,origImgInds] = preparePredictionData_fra(conf,imgData,params,training_phase)
conf.detection.params.detect_min_scale = 1;
if (nargin < 4)
    training_phase = 1;
end

conf.features.winsize = params.wSize;
conf.detection.params.init_params.sbin = params.cellSize;
scaleToPerson = params.scaleToPerson;
img_h = params.img_h; % we set the image height to a constant size to become somewhat scale invariant,
Xs = {};
Fs = {};
offsets = {};
all_boxes = {}; % locations of boxes in each image
imgInds = {}; % index of image in output image set.
origImgInds = {}; % index of image in input image set.


subInds = {}; % index of box in relevant image
imgs = {};
all_scales = {};
masks = {};
values = {};
n = 0;

for k = 1:length(imgData)
    k
    for toFlip = [0 1]
%     for toFlip = 0
        
        %         roiBox = imgData(k).faceBox;
        %         roiBox = round(inflatebbox(roiBox,p.extent,'both',false));
        %         [I,I_rect] = getImage(conf,imgData(k));
        roiParams.infScale = params.extent;
        roiParams.absScale = -1;
        roiParams.centerOnMouth = params.centerOnMouth;
        [rois,roiBox,I] = get_rois_fra(conf,imgData(k),roiParams);
%         clf;imagesc2(I); 
%         pause;
        %         I = cropper(I,roiBox);
        scaleFactor = img_h/size(I,1);
        orig_size = size2(I);
        I = imResample(I,scaleFactor);
        curMask = [];
        objBox = [];
        if (~isempty(imgData(k).objects))
            switch (params.objType)
                case 'obj'
%                     zz = mean(imgData(k).objects(1).poly,1);
%                     objBox = [zz zz];                    
                    objBox = pts2Box(imgData(k).objects(1).poly);
                    objPoly = imgData(k).objects(1).poly;
                    objPoly = bsxfun(@minus,objPoly,roiBox([1 2]));
                    objPoly = objPoly*scaleFactor;
                    curMask = poly2mask2(objPoly,size2(I));                                        
                case 'head';
                    objBox = imgData(k).faceBox;
                case 'hand'
                    handBoxes = imgData(k).hands;
                    objBox = handBoxes(1:min(size(handBoxes,1),1),:); % suffice with one hand
                case 'mouth'
                    mouthBoxes = [imgData(k).mouth imgData(k).mouth];
                    objBox = mouthBoxes(1:min(size(mouthBoxes,1),1),:); % suffice with one hand
            end
            if (isempty(objBox))
                continue;
            end
            objBox = objBox-roiBox([1 2 1 2]);
            objBox = objBox*scaleFactor;
            if (isempty(curMask))
                curMask = poly2mask2(box2Pts(objBox),size2(I));
            end
                
        else
            warning(['no objects of found for image ' imgData(k).imageID]);
            continue;
        end
        n = n+1;
        if (toFlip)
            I = flip_image(I);
            objBox = flip_box(objBox,size2(I));
            curMask = fliplr(curMask);
        end
        masks{end+1} = curMask;
        imgs{end+1} = im2uint8(I);
        %[X,uus,vvs,scales,~,boxes ] = allFeatures( conf,I,1 );
        if (strcmp(params.featType,'hog'))
            [X,uus,vvs,scales,~,boxes ] = allFeatures( conf,I,1 );
        else
            %[F,X] = vl_dsift(im2single(rgb2gray(I)),'step',4,'FloatDescriptors');
            [F,X] = vl_phow(im2single(I),'Step',2,'FloatDescriptors','true','Fast',true,'Sizes',4,'Color','gray');
            X = rootsift(X);
            F = F(:,:);
%             boxes = inflatebbox([F;F]',[12 12],'both',true);
            boxes = inflatebbox([F(1:2,:);F(1:2,:)]',[12 12],'both',true);
            scales = ones(size(X,2),1);
        end
        
        
        if (params.onlyObjectFeatures)
            inds = sub2ind2(size2(I),round(F([2 1],:)'));
            sel_ = curMask(inds);
            F = F(:,sel_);
            X = X(:,sel_);
        elseif (params.coarseToFine)
            xy = boxCenters(boxes);
            %[XX,YY] = meshgrid(1:size(I,2),1:size(I,1));            
%             Z = zeros(size2(I));
%             mm = round(mean(box2Pts(objBox)));
%             Z(mm(2),mm(1)) =1;
%             bw = exp(-bwdist(Z)/10);
            %bw = exp(-bwdist(poly2mask2(round(box2Pts(objBox)),size2(I)))/10);            
            bw = exp(-bwdist(curMask)/10);            
            M = gradientMag(im2single(I));
            bw = bw.*M;            
%             bw = double(bw > .9);                        
            inds = sub2ind2(size2(I),round(F([2 1],:)'));                        
            %si = weightedSample(1:length(inds), bw(inds), min(200,length(inds)));
            si = weightedSample(1:length(inds), bw(inds), min(100,length(inds)));
%             si = si(si>0);            
            F = F(:,si);
            X = X(:,si);
            boxes = inflatebbox([F(1:2,:);F(1:2,:)]',[12 12],'both',true);
            scales = ones(size(X,2),1);
            
            %             figure,imagesc2(I); plotPolygons(F([1 2],:)','g.')
            
            %             figure,imagesc2(I); hold on; plotPolygons(xy(si,:),'g+')
            %             theCenter = boxCenters(objBox);
            
        end
        
        %         figure,imagesc2(I); hold on; plotPolygons(F','r.')
        
        
        
        
        ovps = boxesOverlap(boxes,objBox);
        bc_obj = boxCenters(objBox);
        all_boxes{end+1} = boxes;
        Xs{end+1} = X;
        Fs{end+1} = F;
        values{end+1} = ovps;
        subInds{end+1} = col(1:size(boxes,1));
        imgInds{end+1} = n*ones(size(X,2),1);
        origImgInds{end+1} = imgData(k).imgIndex*ones(size(X,2),1);
        offsets{end+1} = bsxfun(@minus,bc_obj,boxCenters(boxes));
        
        %% debugging
        %         sel_ = 1:31:size(boxes,1);
        %         curOffsets = offsets{end}(sel_,:);
        %         clf;imagesc2(I); hold on; plotBoxes(boxes(sel_,:),'g--');
        %         xy_start = boxCenters(boxes(sel_,:));
        %         plotPolygons(xy_start,'r+');
        %         quiver(xy_start(:,1),xy_start(:,2),curOffsets(:,1),curOffsets(:,2),0,'r');
        %         pause;
        % %
        
        %%
        all_scales{end+1} = scales(:);
    end
end

XX = cat(2,Xs{:});
FF = cat(2,Fs{:});
offsets = cat(1,offsets{:});
all_scales = cat(1,all_scales{:});
imgInds = cat(1,imgInds{:});
origImgInds = cat(1,origImgInds{:});
subInds = cat(1,subInds{:});
values = cat(1,values{:});
% kdtree = vl_kdtreebuild(XX,'Distance','L1');
kdtree = vl_kdtreebuild(XX,'Distance','L2');