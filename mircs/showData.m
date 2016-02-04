
% function [ output_args ] = showData( sampleData)
% %UNTITLED2 Summary of this function goes here
% %   Detailed explanation goes here
% 
% 
% end
% 
[r,ir] = sort(sampleData1.ovps,'ascend');
inds = sampleData1.inds_s;
for it = 1:20:length(ir)
    it
    k = ir(it);
    if r(it) > .1,continue,end
%     if r(it)~=4,continue,end
    curImg = sampleData1.imgs{inds(k)};
    clf;
    displayRegions(curImg,sampleData1.regions(k),[],'dontPause',true);
    title(num2str(r(it)));
    dpc
    
end