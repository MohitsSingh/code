function landmarks_new_parallel(baseDir,d,indRange,outDir,tofix)

cd ~/code/mircs;
addpath('~/code/3rdparty/face-release1.0-basic/'); % zhu & ramanan
initpath;
config;
conf.get_full_image = false; % relevant just in face of stanford40 images where person bounding box is provided.
for k = 1:length(indRange)
    currentID = d(indRange(k)).name;
    [pathstr,name,ext] = fileparts(fullfile(baseDir,currentID));
    resFileName = fullfile(outDir,[name '.mat']);
    if (exist(resFileName,'file'))
        continue;
    end
    
    %faceResFile = fullfile(outDir,[name ,'.mat']);
    faceResFile = fullfile('~/storage/faces_s40_big',[name ,'.mat']);
    load(faceResFile); %-->res.
    conf.imgDir = baseDir;
    [ I ]  = getImage(conf,currentID);
    
    landmarks = struct('results',{},'dpmRect',{},'model',{});
    q = 0;
    for iModel = 1:length(res)
        ds = res(iModel).ds;
        if (isempty(ds))
            continue;
        end
        ds = ds(1,:);
        ds(:,1:4) = clip_to_image(inflatebbox(ds(:,1:4),[1.2 1.2],'both',false),I);
        for iDet = 1:size(ds,1)
            % slightly enlarge the rectangle.
            %ds(1:4) = clip_to_image(inflatebbox(dss,[1.1 1.1],'both',false));
            I_sub = cropper(I,round(ds(iDet,1:4)));
            resizeFactor = 192/(size(I_sub,1));
            I_sub = imresize(I_sub,resizeFactor,'bilinear');
            % pass candidate window to zhu & ramanan detector
            landmarks_results = detect_landmarks_full(conf,{I_sub},1,false);
            landmarks_results = landmarks_results{1};
            for iRes = 1:length(landmarks_results)
                landmarks_results(iRes).xy = landmarks_results(iRes).xy/resizeFactor;
            end
            q = q+1;
            landmarks(q).results = landmarks_results;
            landmarks(q).dpmRect = ds(iDet,:);
            landmarks(q).model = iModel;
        end
    end
    
    save(resFileName,'landmarks');
end
end
