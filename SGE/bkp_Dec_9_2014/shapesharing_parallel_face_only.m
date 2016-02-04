
function shapesharing_parallel_face_only(baseDir,d,indRange,outDir)
restoredefaultpath;
shapeSharingPath = '/net/mraid11/export/data/amirro/shapesharing';
cd(shapeSharingPath);
SetupPath;
addpath(genpath('~/code/utils'));

load ~/storage/misc/face_images_1.mat

for k = 1:length(indRange)
    currentID=d(indRange(k)).name;
    fprintf(1,'%s...',currentID);
%     imgPath = fullfile(baseDir,currentID);
    resPath = j2m(outDir,currentID);
    imgPath = face_images_1{indRange(k)}; %#ok<USENS>
    loadOrCalc([],@get_shape_sharing,imgPath,resPath,[]);
    fprintf(1,'done\n');
end
fprintf('\n\n ***** finished all files in batch ****\n\n\n\n');
function res = get_shape_sharing(conf,imgPath,d) %#ok<INUSL>
res = struct('masks','timing');
[res.masks, res.timing] = ComputeSegment(imgPath);