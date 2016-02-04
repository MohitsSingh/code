function [regions,labels,ovps] = coarseSampleRegions(regions,gt_regions,params)
%     params.learning.ovpType = 'intersection';
    [regions,labels,ovps] = sampleRegions(regions,{gt_regions},params);
%     regions(ovps==1) = [];
%     labels(ovps==1) = [];
end