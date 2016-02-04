function [det,fppi,prec] = compute_det_fppi(detrect,detscore,thres,gtbound,binaryflag,frac)
% function [det,fppi,prec] = compute_det_fppi(detrect,detscore,thres,gtbound,binaryflag,frac)
%
% This function inputs ground truth object bounding boxes, predicted
% bounding boxes and their scores, and output the detection rate, false
% positive per image rate, and precision rate.
%
% Copyright @ Chunhui Gu, April 2009

if nargin < 6,
    frac = 0.2;
end;

assert(length(detrect) == length(gtbound));
assert(length(detrect) == length(binaryflag));

nimgs = length(binaryflag);
for imgId = 1:nimgs,
    detrect{imgId} = detrect{imgId}(detscore{imgId}>=thres,:);
end;

det = 0; fppi = 0; ngt = 0;
nimgs = length(binaryflag);
for imgId = 1:nimgs,
    
    gtbd = gtbound{imgId};
    detbd = detrect{imgId};
    detbd = [detbd(:,1:2) detbd(:,1:2)+detbd(:,3:4)];
    
    if binaryflag(imgId),
        int = zeros(size(gtbd,1),size(detbd,1));
        uni = zeros(size(gtbd,1),size(detbd,1));
        for gtId = 1:size(gtbd,1)
            for detId = 1:size(detbd,1)
                maxx = max(gtbd(gtId,1),detbd(detId,1));
                maxy = max(gtbd(gtId,2),detbd(detId,2));
                minx = min(gtbd(gtId,3),detbd(detId,3));
                miny = min(gtbd(gtId,4),detbd(detId,4));
                if maxx < minx && maxy < miny,
                    int(gtId,detId) = (minx-maxx)*(miny-maxy);
                else
                    int(gtId,detId) = 0;
                end;
                    uni(gtId,detId) = (gtbd(gtId,3)-gtbd(gtId,1)) ...
                                    * (gtbd(gtId,4)-gtbd(gtId,2)) ...
                                    + (detbd(detId,3)-detbd(detId,1)) ...
                                    * (detbd(detId,4)-detbd(detId,2)) ...
                                    - int(gtId,detId);
            end;
        end;
        i2u = int ./ uni;
        [i2u_max,I] = max(i2u,[],1);
        nhits = length(unique(I(i2u_max>=frac)));
        
        detflag = false(1,size(detbd,1));
        uniq = unique(I);
        for ii = 1:length(uniq),
            a = find(I==uniq(ii));
            [i2u_mmax, II] = max(i2u_max(a));
            if i2u_mmax >= frac,
                detflag(a(II)) = true;
            end;
        end;
        
        det = det + nhits;
        fppi = fppi + size(detbd,1) - nhits;
        ngt = ngt + size(gtbd,1);
    else
        fppi = fppi + size(detbd,1);
        detflag = false(1,size(detbd,1));
    end;
    
%     if isvisualize && binaryflag(imgId),
%         img = imread(test_name{imgId});
%         figure(1); clf;
%         subplot(2,2,1); imshow(img,[]);
%         for re = 1:size(detbd,1),
%             rectangle('Position',[detbd(re,1:2) detbd(re,3:4)-detbd(re,1:2)],'EdgeColor','g','LineWidth',2);
%         end;
%         subplot(2,2,2); imshow(img,[]);
%         for re = 1:size(gtbd,1),
%             rectangle('Position',[gtbd(re,1:2) gtbd(re,3:4)-gtbd(re,1:2)],'EdgeColor',[1 0.5 0.5],'LineWidth',2);
%         end;
%         subplot(2,2,3); imshow(img,[]);
%         for re = 1:size(detbd,1),
%             if detflag(re),
%                 rectangle('Position',[detbd(re,1:2) detbd(re,3:4)-detbd(re,1:2)],'EdgeColor','g','LineWidth',2);
%             end;
%         end;
%         subplot(2,2,4); imshow(img,[]);
%         for re = 1:size(detbd,1),
%             if ~detflag(re),
%                 rectangle('Position',[detbd(re,1:2) detbd(re,3:4)-detbd(re,1:2)],'EdgeColor','r','LineWidth',2);
%             end;
%         end;
%         keyboard;
%     end;
    
end;
if det+fppi == 0,
    prec = NaN;
else
    prec = det / (det+fppi);    %% precision
end;
det = det / ngt;            %% recall (detection)
fppi = fppi / nimgs;        %% False positive per image