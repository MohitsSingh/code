function frs = filterMultiscale(I,fb)
I = im2single(I);

% fb = FbMake(2,1,0);
% run gabor filters at multiple scales.

%fb = fb(:,:,1:end-3);
minSize = size(fb,1);

% I = rgb2gray(strawImage);

scales = 2.^(0:-2:log2(minSize/size(I,1)));
frs = [];
for iScale = 1:length(scales)
    I_scale = imResample(I,scales(iScale));
    fr = FbApply2d(I_scale,fb,'same',0);
    fr = imResample(fr,dsize(I,1:2),'nearest');
    if (iScale==1)
        frs = (fr);
    else
        frs = max(frs,fr);
    end
    
%     montage2(fr);
%     fr = sum(fr.^2,3);%sum(abs(fr),3);
%     clf;subplot(1,2,1); imagesc(I); axis image;
%     H = fhog(I_scale,4);
%     fr = hogDraw(H.^2,15,1);
%     subplot(1,2,2); imagesc(fr); axis image;
%     clf;imagesc(fr);
%     pause;
end

ddd = 3;
frs(1:ddd,:,:) = 0;
frs(end-ddd+1:end,:,:) = 0;

frs(:,1:ddd,:) = 0;
frs(:,end-ddd+1:end,:) = 0;


% subplot(2,1,1); imagesc(I); axis image
% subplot(2,1,2); montage2(abs(frs),struct('extraInfo',1));
% pause
%      2.^(log(minSize):log(size(I,1)))
%      scales = minSize([1:.5:size(I,1)/minSize)