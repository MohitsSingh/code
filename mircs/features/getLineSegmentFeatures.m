function lineSegments = getLineSegmentFeatures(conf,imgData,I)
% load line features from line-segment calculation and ELSD calculation
    [lines_,ellipses_] = getELSDResults(conf,imgData.imageID);
    
    lineSegments = lines_(:,1:4);
    
%     E = edge(im2double(rgb2gray(I)),'canny');
%     [edgelist edgeim] = edgelink(E, []);
%     segs = lineseg(edgelist,3);
    % compute line segments on the fly instead of loading.
    
%     lineSegFile = j2m(conf.lineSegDir,imgData);
%     L = load(lineSegFile);
%     segs = seglist2segs(L.seglist);    
%     lineSegments = [segs(:,[2 1 4 3]);lineSegments];
    
    
    
    
    
% get line
end