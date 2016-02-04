function landmarks_new_parallel_ims(baseDir,d,indRange,outDir,tofix)

cd ~/code/mircs;
addpath('~/code/3rdparty/face-release1.0-basic/'); % zhu & ramanan
initpath;
config;
conf.get_full_image = false; % relevant just in face of stanford40 images where person bounding box is provided.
% L_imgs = load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat');
% d = L_imgs.ims;
for k = 1:length(indRange)
    k
    I = d(indRange(k)).img;
    resFileName = fullfile(outDir,sprintf('%05.0f.mat',indRange(k)));
    if (exist(resFileName,'file'))
        continue;
    end
    
    resizeFactor = 2;
    landmarks = detect_landmarks_99(conf,{I},resizeFactor,false);
    landmarks = landmarks{1};
    for iRes = 1:length(landmarks)
        landmarks(iRes).xy = landmarks(iRes).xy/resizeFactor;
    end
    save(resFileName,'landmarks');
end
disp('DONE');
end
