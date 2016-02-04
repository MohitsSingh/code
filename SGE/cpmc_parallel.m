function allres = cpmc_parallel(imagePaths,inds,jobID,outDir,extraInfo,job_suffix)
cd ~/code/mircs/;
conf = extraInfo.conf;
path(extraInfo.path);
newImageData = extraInfo.newImageData;
if (~isfield(extraInfo,'minFaceScore'))
    minFaceScore = -.6;
else
    minFaceScore = extraInfo.minFaceScore;
end

for k = 1:length(inds)
    newImageData(inds(k)).imagePath = getImagePath(conf,newImageData(inds(k)).imageID);
end

cd /home/amirro/code/3rdparty/cpmc_release1/;
addpath('./code/');
addpath('./external_code/');
addpath('./external_code/paraFmex/');
addpath('./external_code/imrender/vgg/');
addpath('./external_code/immerge/');
addpath('./external_code/color_sift/');
addpath('./external_code/vlfeats/toolbox/kmeans/');
addpath('./external_code/vlfeats/toolbox/kmeans/');
addpath('./external_code/vlfeats/toolbox/mex/mexa64/');
addpath('./external_code/vlfeats/toolbox/mex/mexglx/');
addpath('./external_code/globalPb/lib/');
addpath('./external_code/mpi-chi2-v1_5/');
for k = 1:length(inds)
    k
    res = struct;
    res(1).masks = [];
    res(1).scores = [];
    curImageData = newImageData(inds(k));
    if (curImageData.faceScore >= minFaceScore)
        resPath = fullfile(outDir,strrep(curImageData.imageID,'.jpg',['_' job_suffix '.mat']));
        if (~exist(resPath,'file'))
            fprintf('calculating for image %s!\n',curImageData.imageID);
            I = imread(newImageData(inds(k)).imagePath);
            %         I = imresize(I,[64 NaN],'bilinear');
            %tmp_name = char(java.util.UUID.randomUUID);
            [~,fName,~] = fileparts(curImageData.imageID);
            %fName = [num2str(inds(k)) tmp_name];
            if (isfield(extraInfo,'runMode'))
                fName = [fName '_' extraInfo.runMode];
                if strcmp(extraInfo.runMode,'upperBody')
                    rects = curImageData.upperBodyDets;
                    if (~isempty(rects))
                        I = cropper(im2uint8(I),round(rects(1,1:4)));
                        res(1).rect = round(rects(1,1:4));
                    end
                elseif strcmp(extraInfo.runMode,'sub')
                    [I,~,face_box] = getSubImage(conf,curImageData,2);
                    res(1).rect = round(face_box);
                end
            end
            
            % clear the CPMC dir
            clearCPMC('',fName);
            imwrite(I,['data/JPEGImages/' fName '.jpg']);
            [masks, scores] = cpmc('data/', fName);
            res(1).masks = masks;
            res(1).scores = scores;
            
            save(resPath,'res');
            clearCPMC('',fName); % cleanup.
        else
            load(resPath);
        end
    end
    %     end
    allres{k} = res;
    fprintf('done with image %s!\n',curImageData.imageID);
end
fprintf('***********************************\n************FINISHED***************\n***********************************\n');

res = allres;
resFileName = fullfile(outDir,sprintf('job_%05.0f_%s_agg.mat',jobID,job_suffix));
save(resFileName,'res','inds','jobID');