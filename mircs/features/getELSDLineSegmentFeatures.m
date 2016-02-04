function lineSegments = getELSDLineSegmentFeatures(conf,imgData,outPath)
% load line features from line-segment calculation and ELSD calculation
    if (nargin < 3)
        [lines_,ellipses_] = getELSDResults(conf,imgData.imageID);    
    else
        [lines_,ellipses_] = getELSDResults(conf,imgData.imageID,outPath);    
    end
    lineSegments = lines_(:,1:4);
                    
% get line
end