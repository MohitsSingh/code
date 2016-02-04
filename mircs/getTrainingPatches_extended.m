function trainingData = getTrainingPatches_extended(inputDir)


trainingData = struct('img',{},'obj_rect',{},'img_data',{});
% allObjTypes = {'cup','bottle','straw'};
allObjTypes = {'straw'};
t = 0;
for iObjectType = 1:length(allObjTypes)
    objTypes = (allObjTypes(iObjectType));
    imgDir  = fullfile(inputDir,allObjTypes{iObjectType});
    resDir = fullfile(imgDir,'annotations');
%     resDir ='/home/amirro/data/drinking_extended/straw/straw_anno'
    if (~exist('resDir','dir')), mkdir(resDir);end
    d = dir(fullfile(imgDir,'*.jpg'));
    for k = 1:length(d)    
        k
        fName = fullfile(resDir,[d(k).name '.txt']);
        [objs,bbs] = bbGt( 'bbLoad', fName);
%         bbs
% %         continue
        bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
        bb = bbs(1:4);
        %     end
        imgPath = fullfile(imgDir,d(k).name);
        I = imread(imgPath);
        
%         clf,imagesc(I); hold on; axis image; plotBoxes(bb); 
        bb = inflatebbox(bb,[1.5 1.5],'both',false);        
        bb = round(makeSquare(bb));
%         plotBoxes(bb,'g');
%         pause; continue;
        
        bb_large = round(inflatebbox(bb,[2 2],'both',false));
% %         
% %         clf; imagesc(I); axis image; hold on; plotBoxes(bb_large,'r--','LineWidth',2);
% %         plotBoxes(bb,'g','LineWidth',2);pause; continue;
% %      
        %img = cropper(I,bb);
%         img = I;
%         bb = bb-bb_large([1 2 1 2]);
        
        t = t+1;
        trainingData(t).img = I;
        trainingData(t).obj_rect = bb;
        
        trainingData(t).img_data.imageID = imgPath;
        clf; imagesc(I); axis image; hold on; plotBoxes(bb,'g--','LineWidth',2);pause; continue;                
    end   
end