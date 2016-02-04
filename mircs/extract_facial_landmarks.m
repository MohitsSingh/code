function [xy_global, xy_local, faceBox, pose, I, resizeFactor] = extract_facial_landmarks(I_orig,curFaceDet,facialLandmarkData)
%EXTRACT_FACIAL_LANDMARKS Use pre-trained models to extract faciallandmarks

regModel_frontal = facialLandmarkData.regModel_frontal;
regModel_profile = facialLandmarkData.regModel_profile;
regPrm_frontal= facialLandmarkData.regPrm_frontal;
regPrm_profile = facialLandmarkData.regPrm_profile;
prunePrm = facialLandmarkData.prunePrm;
nFrontal = facialLandmarkData.nFrontal;
poseMap = [90 -90 30 -30 0 0];
d_size = 80;
faceBox = curFaceDet.boxes_rot(1,:);
pose = poseMap(faceBox(5));
conf.get_full_image = true;
curScore = faceBox(6);
curRot = faceBox(7);
fprintf('comp: %d,pose: %d\n',faceBox(5),pose);
I = cropper(imrotate(I_orig,curRot,'bilinear','crop'),round(faceBox));
resizeFactor = d_size/size(I,1);
I = imResample(I,resizeFactor);
frontal = abs(pose)~=90;
needFlip = false;
nn_initialization = true;

if (~frontal && pose==90)
    needFlip = true;
    %
end

if (pose==30)
    needFlip = true;
    I = flip_image(I);
end

if (needFlip)
    I =  flip_image(I);
end
beVerbose = 0;
ff = [1 1 fliplr(size2(I)-1)];

RT1 = facialLandmarkData.RT1;
my_inds = [];
if (frontal)
    if nn_initialization
        my_inds = find(facialLandmarkData.all_c>3);
        my_inds = vl_colsubset(my_inds,RT1,'random');
        RT1_1 = min(RT1,length(my_inds));
    end
    p=shapeGt('initTest',{I},ff,regModel_frontal.model,...
        regModel_frontal.pStar,regModel_frontal.pGtN,RT1_1,my_inds);
    testPrm = struct('RT1',RT1_1,'pInit',ff,...
        'regPrm',regPrm_frontal,'initData',p,'prunePrm',prunePrm,...
        'verbose',beVerbose);
    t=clock;[p,pRT] = rcprTest({I},regModel_frontal,testPrm);t=etime(clock,t);
else
    if nn_initialization
        my_inds = find(facialLandmarkData.all_c<=3)-nFrontal;
        my_inds = vl_colsubset(my_inds,RT1,'random');
        RT1_1 = min(RT1,length(my_inds));
    end
    p_init=shapeGt('initTest',{I},ff,regModel_profile.model,...
        regModel_profile.pStar,regModel_profile.pGtN,RT1_1,my_inds);
    testPrm = struct('RT1',RT1_1,'pInit',ff,...
        'regPrm',regPrm_profile,'initData',p_init,'prunePrm',prunePrm,...
        'verbose',beVerbose);
    t=clock;[p,pRT] = rcprTest({I},regModel_profile,testPrm);t=etime(clock,t);
end
xy_local = [p(1:end/2);p(end/2+1:end)]';
% get points in original image coordinate system.
xy_global = xy_local;
if (needFlip)
    xy_global = flip_pt(xy_global,size2(I));
end
xy_global =  bsxfun(@plus,faceBox(1:2),xy_global/resizeFactor);
xy_global = rotate_pts(xy_global,-pi*curRot/180,fliplr(size2(I_orig)/2));
end

