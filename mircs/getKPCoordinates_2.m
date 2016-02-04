function kps = getKPCoordinates_2(ptsData,requiredKeypoints)
    kps = NaN(length(ptsData),length(requiredKeypoints),2);        
    for t = 1:length(ptsData)
        
%         curIm = imread(paths{t});
%         curPts =   bsxfun(@minus,ptsData(t).pts,ress(t,1:2));
        
        % I can do this using a hashtable but it's not important.                        
        for iReqKeyPoint = 1:length(requiredKeypoints)        
            ii = strcmp(requiredKeypoints{iReqKeyPoint}, ptsData(t).pointNames);
            if (none(ii))
                kps(t,iReqKeyPoint,:) = NaN;
            else                
                 kps(t,iReqKeyPoint,:) = ptsData(t).pts(ii,:);
            end        
        end
        %clf; imagesc2(curIm); plotPolygons(curPts,'g.')
    end
end
