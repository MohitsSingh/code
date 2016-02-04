function rois = sampleAround(cur_config,nSamples,scaleFactor,params,I,constrainAngle)
if strcmp(params.cand_mode,'boxes')
    rois = getCandidateRoisBoxes(cur_config.bbox,scaleFactor,params.sampling,I);
    
    if (constrainAngle)
        ad = 180*abs(angleDiff(pi*cellfun3(@(x) x.theta,rois)/180,pi*cur_config.theta/180))/pi;
        goods = ad <= params.sampling.maxThetaDiff;
        rois = rois(goods);
    end
    
    rois = vl_colsubset(row(rois),nSamples,'Uniform');
    rois = [rois{:}];
    for t = 1:length(rois)
        rois(t).bbox = pts2Box(rois(t).xy);
        rois(t).ispoly = false;
    end
    
    %     poly_masks = cellfun2(@(x) poly2mask2(x.xy,size2(I)), rois);
    %         displayRegions(I,poly_masks);
    %     cur_config(iNode).endPoint = bestRoi.endPoint;
    %     cur_config(iNode).mask = poly_masks{ir(1)};
    %     cur_config(iNode).bbox = region2Box(cur_config(iNode).mask);
    %     cur_config(iNode).theta = bestRoi.theta;
    %     cur_config(iNode).poly = bestRoi.xy;
else
    error('current candidate mode not supported')
end