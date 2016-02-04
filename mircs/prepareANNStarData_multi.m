function [features,offsets] = prepareANNStarData_multi(conf,imgs,pts,params)
% pts is 
Xs = {};
% Fs = {};
offsets = {};
if (nargin < 4)
    params = struct('maxFeats',200);
end
% imgInds = {}; % index of image in output image set.
% origImgInds = {}; % index of image in input image set.
% subInds = {}; % index of feature in relevant image
% masks = {};
% values = {};
n = 0;
if (~iscell(pts))
    pts = mat2cell2(pts,[size(pts,1),1]);
end
for k = 1:length(imgs)
    %     k
    %     for toFlip = [0 1]
    for toFlip = 0
        I = imgs{k};
        curPt = round(pts{k});
        n = n+1;
        if (toFlip)
            I = flip_image(I);
            curPt = flip_pt(curPt,size2(I));
        end
        %[F,X] = vl_phow(im2single(I),'Step',1,'FloatDescriptors','true','Fast',true,'Sizes',4,'Color','gray');
        
        curMask = zeros(size2(I));
        curMask(curPt(2),curPt(1)) = 1;
        %bw = exp(-bwdist(curMask)/10);
        distImage=bwdist(curMask);
        bw = exp(-distImage/100);
        
        
        %         [y_in,x_in] = find(bw);
        
        
        [F,X] = vl_phow(im2single(I),'Step',1,'FloatDescriptors','true','Fast',true,'Sizes',4,'Color','gray');
        s = sum(X,1);
        X(:,s==0) = [];
        F(:,s==0) = [];
        X = rootsift(X);
        boxes = inflatebbox([F(1:2,:);F(1:2,:)]',[12 12],'both',true);
        %         xy = boxCenters(boxes);
        
        %         bw = ones(size(bw));
        M = gradientMag(im2single(I));
        bw = bw.*M;
        bw(distImage>params.cutoff) = 0;
        %         bw = M;
        %         bw = M;
        inds = sub2ind2(size2(I),round(F([2 1],:)'));
        si = weightedSample(1:length(inds), bw(inds), min(params.maxFeats,length(inds)));
        si = si(si > 0);
        F = F(:,si);
        X = X(:,si);
        boxes = boxes(si,:);
        %             figure,imagesc2(I); plotPolygons(F([1 2],:)','g.')
        bc_obj = curPt;
        Xs{end+1} = X;
        curOffsets = bsxfun(@minus,bc_obj,boxCenters(boxes));
        offsets{end+1} = curOffsets;
%                 figure,imagesc2(I); plotPolygons(F([1 2],:)','g.');
%                 plotPolygons(curPt,'rd');
%                 quiver(F(1,:)',F(2,:)',curOffsets(:,1),curOffsets(:,2),0);
        
    end
end
features = cat(2,Xs{:});
offsets = cat(1,offsets{:});
