function piotr_faces_big_parallel(baseDir,d,indRange,outDir)
cd ~/code/mircs;
initpath;
config;
load ~/storage/misc/imageData_new;
load ~/mircs/experiments/experiment_0032/detectors_big_full.mat
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
    I = getImage(conf,curImageData.imageID);
    all_bbs = {};
    rescaleFactor = 2;
    I_orig = imResample(I,rescaleFactor);
    for rots = -40:10:40
        I = imrotate(I_orig,rots,'bilinear','crop');
        bb = acfDetect(I,detectors);
        if (~isempty(bb))
            bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
            %             clf; imagesc(normalise(I)); axis image; hold on; plotBoxes(bb,'g','LineWidth',2);
            %             drawnow; pause;
            bb(:,1:4) = bb(:,1:4)/rescaleFactor;
            bb(:,end) = rots;
            all_bbs{end+1} = bb;
        end
        
    end
    bbs = cat(1,all_bbs{:});
    if (isempty(bbs))
        bbs = zeros(0,7);
    end
    save(resFileName,'bbs');
    fprintf('\tdone witn current image!\n');
end
fprintf('\n\nFinished all images in current batch\n\n\n!\n');

