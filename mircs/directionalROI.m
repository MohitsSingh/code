function roi = directionalROI(im,startPt,vec,roiWidth)
    if (nargin < 4)
        roiWidth = 10;
    end
    [X,Y] = meshgrid(1:size(im,2),1:size(im,1));
    
    X_ = X-startPt(1);
    Y_ = Y-startPt(2);
    xy_vecs = [X_(:),Y_(:)]';
    [xy_vecs,norms] = normalize_vec(xy_vecs);
    vec = vec/norm(vec);
    dots = vec'*xy_vecs;
    dots = reshape(dots,size(X));
    T_angle = 90-roiWidth;
    
    % use line seg.
      
    roi = dots>=sind(T_angle);
    roi = imdilate(roi,ones(7));
    

%     imshow(repmat(roi,[1 1 3]).*im)
%     
%     
%     imshow(roi.*M,[]);
%         
%     imshow(repmat(roi,[1 1 3]).*im);
  
    
end