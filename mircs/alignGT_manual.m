function groundTruth = alignGT_manual(conf,groundTruth,objectName)
resPath=  fullfile(conf.cachedir,[objectName '_aligned.mat']);
if (exist(resPath,'file'))
    load(resPath)
    return;
end
gtParts = {groundTruth.name};
isObj = cellfun(@any,strfind(gtParts,objectName));
groundTruth = groundTruth(isObj);
sourceImage = '';
objImages = {};
for k = 1:length(groundTruth)
    curGT = groundTruth(k);   
        if (~strcmp(curGT.sourceImage,sourceImage))
            I = getImage(conf,curGT.sourceImage);
            sourceImage = curGT.sourceImage;
        end
        z = false(dsize(I,1:2));
        bbox = [ pts2Box([curGT.polygon.x,...
            curGT.polygon.y]) curGT.Orientation];
        
        I_cropped = cropper(I,round(inflatebbox(bbox,[1.5,1.5],'both')));
        curTheta = 0;
        groundTruth(k).curTheta = 0;
        disp('arrows to change rotation, w to abort, any other key for next image');
        while (true)
            imRotated = max(0,min(1,im2double(imrotate(I_cropped,curTheta','bilinear','crop'))));
            clf; imagesc(imRotated);axis image;
            
            t = getkey;
            dTheta = 15;
            if (t==28) % left
                curTheta = curTheta + dTheta;
            elseif (t==29)
                curTheta = curTheta - dTheta;                
            elseif (t==119)
                disp('aborting - nothing is saved');
                close all; 
                return;
            else
                disp(['final angle = ' num2str(curTheta)]);
                break;                
            end
        end
        groundTruth(k).curTheta = curTheta;
%         bbox(1:4) = round(bbox(1:4));
%         z(bbox(2):bbox(4),bbox(1):bbox(3)) = 1;
%         objImages{end+1} =  I(bbox(2):bbox(4),bbox(1):bbox(3),:);        
end
save(resPath,'groundTruth');