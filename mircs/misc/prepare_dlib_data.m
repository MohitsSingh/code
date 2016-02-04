function prepare_dlib_data(conf,fra_db,outPath)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if nargin < 3
    outPath = 'out.txt';
end
fid = fopen('out.txt','w+');
for t = 1:length(fra_db)
    if (mod(t,100)==0)
        t
    end
    curImgData = fra_db(t);
    %if (curImgData.classID~= conf.class_enum.BRUSHING_TEETH),continue,end
    detections = curImgData.raw_faceDetections.boxes(1,:);
    [I_orig,I_rect] = getImage(conf,curImgData,[],false);
    conf.get_full_image = true;
    faceBox = round(detections(1:4));
    faceBox = faceBox + I_rect([1 2 1 2]);
    clf; imagesc2(I_orig);plotBoxes(faceBox);dpc;continue
    fprintf(fid,['D:/datasets/stanford40/JPEGImages/',curImgData.imageID '\n']);
    fprintf(fid,'1\n');
    fprintf(fid,'%d %d %d %d\n',faceBox(1),faceBox(2),faceBox(3)-faceBox(1),faceBox(4)-faceBox(2));
    %     clf; imagesc2(I_orig); plotBoxes(faceBox);
    %     pause
end
fclose(fid)

end

