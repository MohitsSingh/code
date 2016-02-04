
function regions = getGCRegions(conf,imgData,thetaRange)
regions = {};close all;
[im,xmin,xmax,ymin,ymax] = getImage(conf,imgData.imageID);
[M,~,face_box,face_poly] = getSubImage(conf,imgData,2.5,true);
orig_sz = size2(M);
T = 1.5;
if (nargin < 3)
    thetaRange = 0:10:350
end
for theta = thetaRange
    %             theta
    M = imresize(M,[128 NaN]);
    sz = size2(M);
    ss = mean(sz);
    [A,center,majorAxis,minorAxis] = directionalROI_rect(M,sz/2,theta,ss/3);
%     mm = maskEllipse(size(M,1),size(M,2),center(2),center(1),majorAxis/4,minorAxis/4,pi*theta/180);    
    mm1 = bwdist(mm);
    mm2 = bwdist(~mm);
    dMask = -mm2+max(mm2(:))+~mm.*mm1;
    
    
    if (1)
    
    G = fspecial('gauss',round([T T]*majorAxis),majorAxis/2);
    % create a gaussian centered at the center of the desired region with
    % major and 
    G = imresize(G,round(T*[minorAxis*.7 1.2*majorAxis]));
    
    G = imrotate(G,-theta,'bilinear');
    %
    X = zeros(size(A));
    X(round(center(2)),round(center(1))) = 1;
    G = imfilter(single(X),G,'same');
    G = double(G/max(G(:)));
    else
        G = double(exp(-dMask/50));
    end
    %    imagesc(G);
    %covmat
    % % %     mm = double(mm);
    % % %     B = bwdist(mm);
    % % %     B1 = (bwdist(~mm)-1).*mm;
    % % %     B = B-B1;B = B-min(B(:));
    % % %     %         imagesc(B);%
    % % %     G = double(exp(-B/(majorAxis/6)));
    %             clf;imagesc(G); drawnow; pause;continue;
    M = clip_to_bounds(M);
    curRegions = getSegments_graphCut(M,G,128,true);    
    rprops = regionprops(curRegions,'PixelIdxList');
    for q = 1:length(rprops)
        z = false(size(G));
        z(rprops(q).PixelIdxList) = true;
        regions{end+1} = z;
    end
%     regions{end+1} = getSegments_graphCut(M,G,128,false);
    %drawnow
    % %             pause
end
regions = cellfun2(@(x) imResample(x,orig_sz,'nearest'),regions);
regions = shiftRegions(regions,round(face_box),im);
% refine all the regions now over the entire image.
% regions = [regions(:),col(cellfun2(@(x)refineRegionGC(im,x,2),regions))];
