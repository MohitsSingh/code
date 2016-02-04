function trainingData = getTrainingPatches(conf,imageData,newImageData,useGT,subset)

if (nargin < 4)
    useGT = true;
end
if (nargin < 5)
    subset = 'train';
end
trainingData = struct('label',{},'face_rect',{},'obj_rect',{},'img_data',{});
if (strcmp(subset,'train'))    
%     imageSet = imageData.train;
else
%     imageSet = imageData.test;
end
groundTruth = consolidateGT(conf,subset,false);
[gt_train,keys] = groupBy(groundTruth,'sourceImage');
conf.get_full_image = true;
posemap = 90:-15:-90;

rects = {};
train_imgs = {};
face_rects = {};
z = 0;
for k = 1:length(gt_train)
    k
    curID =  gt_train(k).key;
    imageIndex = findImageIndex(newImageData,curID);
    curPose = posemap(newImageData(imageIndex).faceLandmarks.c);
    if (~useGT)
        if (newImageData(imageIndex).faceScore < -.6)
            curPose = -1;
        end
    end
    I = getImage(conf,curID);
    curGroup = gt_train(k).group;
    
    found = false;
    facePoly = [];
    objMasks = {};
    mouthPoly = [];
    for kk = 1:length(curGroup)
        curName = curGroup(kk).name;
        p = curGroup(kk).polygon;
        switch(curName)
            case {'cup','bottle','can','straw'};
                objMasks{end+1} = poly2mask2([p.x p.y],size2(I));
            case 'face'
                faceMask = poly2mask2([p.x p.y],size2(I));
            case 'mouth'
                mouthPoly = [p.x p.y];
        end
    end
    train_imgs{k} = I;
    mouthCenter = mean(mouthPoly,1);
    [yy,xx] = find(faceMask);
    faceBox = pts2Box(xx,yy);
    if (~useGT)
        [m,~,faceBox] = getSubImage(conf,newImageData,curID);
    end
    
    face_rects{end+1} = [faceBox k];
    faceScale = faceBox(4)-faceBox(2);
    objMasks = {max(cat(3,objMasks{:}),[],3)};
    for q = 1:length(objMasks)
        int = objMasks{q} & faceMask;
        
        %         d1 = bwdist(objMasks{q}) <= 2;
        %         d2 = bwdist(faceMask) <= 2;
        %         int = d1 & d2;
        %
        if (nnz(int) == 0)
            d1 = bwdist(objMasks{q})+bwdist(faceMask);
            [~,id] = min(d1(:));
            [yy,xx] = ind2sub(size(d1),id);
            
        else
            
            [yy,xx] = find(int);
        end
        curRect = pts2Box(xx,yy);
        
        %         curRect = round([mouthCenter mouthCenter]);
        if (useGT)
            curRect = inflatebbox(curRect,faceScale/1.5,'both',true);
        else
            curRect = inflatebbox(curRect,1.5,'both',false);
        end
        
        
        %         curRect = inflatebbox(curRect,faceScale*1.5,'both',true);
        
        %         curRect = BoxIntersection(curRect,faceBox);
        %         p = curRect(4)-curRect(2);
        %         tt = 40;
        %         if (p < tt)
        %             curRect = inflatebbox(curRect,[tt tt],'both',true);
        %         end
        curRect(:,11) = k;
        rects{end+1} = [curRect curPose];
        
        z = z+1;
        trainingData(z).label = 1;
        trainingData(z).img = I;
        trainingData(z).img_data = newImageData(imageIndex);
        trainingData(z).obj_rect = curRect;
        trainingData(z).face_rect = faceBox;
        %         clf;imagesc(I); hold on; plotBoxes(curRect);
        %         pause
    end
end

rects = cat(1,rects{:});
face_rects = cat(1,face_rects{:});
end