function lips_parallel(baseDir,d,indRange,outDir)
cd ~/code/mircs;
initpath;
config;
load ~/storage/misc/imageData_new;
load ~/storage/misc/imageData_new;
load ~/mircs/experiments/experiment_0033/detectors_full.mat
detectors{1}.opts.pNms.separate=1;

for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    imagePath = fullfile(baseDir,currentID);
    [~,filename,~] = fileparts(imagePath);
    resFileName = fullfile(outDir,[filename '.mat']);
    fprintf('checking if filename %s exists :.... \n',resFileName);
    if (exist(resFileName,'file'))
        fprintf('already exists! skipping ... \n');
        continue;
    end
    fprintf('calculating for: %s!\n',filename);
    
    imageIndex = findImageIndex(newImageData,currentID);
    curImageData = newImageData(imageIndex);
    if (curImageData.faceScore >-1000)
        [M,landmarks,face_box,face_poly] = getSubImage(conf,curImageData,1,true);
        M = imResample(M,[100 100],'bilinear');
        curScore = -inf;
        all_bb = {};
        for rot = -30:10:30
            M_rot = imrotate(M,rot,'bilinear','crop');
            bbs = acfDetect(M_rot,detectors);
            %     bbs = cat(1,bbs{:});
            if (~isempty(bbs))
                curScore = max(curScore,max(bbs(:,5)));
                [b,ib] = sort(bbs(:,5),1,'descend');
                bbs = bbs(ib,:);
                %                 bbs
                bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
                bbs(:,7) = rot;
                %                 clf; imagesc(M_rot); axis image; hold on; plotBoxes(bbs,'g','LineWidth',2);drawnow; pause
            else
                bbs = zeros(0,7);
            end
            all_bb{end+1} = bbs;
        end
        bbs = cat(1,all_bb);
    else
        bbs = zeros(0,7);
    end
    
    save(resFileName,'bbs');
    fprintf('\tdone witn current image!\n');
    
end
fprintf('\n\nFinished all images in current batch\n\n\n!\n');

