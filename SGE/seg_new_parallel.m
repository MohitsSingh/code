function res = seg_new_parallel(initData,params)

if (~isstruct(initData) && strcmp(initData,'init'))
    addpath('~/code/SGE');
    addpath(genpath('~/code/utils'));
    addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox'));
    cd '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';
    install;
    
    load '~/code/mircs/fra_db_2015_10_08.mat';
    %     fra_db = reqInfo.fra_db;
    res.fra_db = fra_db; %#ok<NODEF>
    return
end

fra_db = initData.fra_db;
[~,b,c] = fileparts(params.name);
curName = [b c];%run_all_3(params,outDir,'seg_new_parallel',testing,'mcluster03');

k = findImageIndex(fra_db,curName);
imgData = fra_db(k);
% fra_db_mouth_seg
fprintf(1,'working on image data:%s\n',imgData.imageID);
useGT = [false true];
segs = struct;

if params.full_image
    I = imread(params.name);
    I = imResample(I,.5,'bilinear');
    [candidates, ucm2] = im2mcg(I,'accurate',true);
    segs(1).success = true;
    segs(1).candidates = candidates;
    segs(1).mouthBox = [];
    segs(1).ucm2 = single(ucm2);
    segs(1).usedHalf = true;
else
    for iUseGT = 1:length(useGT)
        I = imread(params.name);
        fprintf(1,'checking if around mouth...\n');
        
        if isfield(params,'aroundMouth') && params.aroundMouth
            fprintf(1,'YES\n');
            if useGT(iUseGT)
                fprintf(1,'using gt..\n');
                faceBox = imgData.faceBox;
                mouthCenter = imgData.landmarks_gt.xy(3,:);
            else
                fprintf(1,'not using gt..\n');
                faceBox = imgData.faceBox_raw;
                mouthCenter = imgData.landmarks.xy(3,:);
            end
            h = faceBox(3)-faceBox(1);
            h = 2*h;
            fprintf(1,'inflating bbox...\n');
            mouthBox = round(inflatebbox(mouthCenter,h,'both',true));
            I = cropper(I,mouthBox);
            fprintf(1,'image has been cropped\n');
        end
        segs(iUseGT).msg = '';
        segs(iUseGT).useGT = useGT(iUseGT);
        try
            fprintf(1,'trying to segment....\n');
            [candidates, ucm2] = im2mcg(I,'accurate',true);
            segs(iUseGT).success = true;
            fprintf(1,'sucess!!\n');
            %     res.msg
        catch e
            segs(iUseGT).success = false;
            %I = imread(params.name);
            sz_orig = size2(I);
            I = imResample(I,2,'bilinear');
            segs(iUseGT).msg = 'failed for sub-image, increasing by factor of 2 and trying again';
            [candidates, ucm2] = im2mcg(I,'accurate',true);
            candidates.masks = imResample(candidates.masks,sz_orig,'nearest');
            %         segs(iUseGT).candidates = candidates;
            %         segs(iUseGT).ucm2 = single(ucm2);
            fprintf(1,'error:....');
            fprintf(1,segs(iUseGT).msg);
        end
        
        segs(iUseGT).candidates = candidates;
        segs(iUseGT).mouthBox = mouthBox;
        segs(iUseGT).ucm2 = single(ucm2);
    end
end
res = struct;
res.segs = segs;