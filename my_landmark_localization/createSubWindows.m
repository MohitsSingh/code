function [sub_windows] = createSubWindows(IsTr,phisTr)
sizes = cell2mat(cellfun2(@size2,IsTr));% height/width
factors = sizes(:,1);
nKP = size(phisTr,2);
% (learn how to estimate the occlusion even if the prediction is
% inaccurate, by moving the center around a bit?)
d1= 5;d2 = 20;
imgToBoxRatio = 3;
sub_windows = zeros([size2(phisTr),5]);
for iKP = 1:nKP
    curKP = squeeze(phisTr(:,iKP,:));
    
    curKP_centers = curKP(:,1:2);
    
    jitters = rand(size(factors,1),2).*[factors factors]/d1-[factors factors]/(2*d1);
    curKP_centers = curKP_centers+jitters;
    curRects = round(inflatebbox([curKP_centers curKP_centers],factors/imgToBoxRatio,'both',true));
    curLabels = curKP(:,3);
    sub_windows(:,iKP,:) = [curRects curLabels];
end

%plotPolygons(jitters,'r.')




end
