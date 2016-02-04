function [lines_,ellipses_] = getELSDResults(conf,currentID,outDir)
if (nargin < 3)
    outDir = conf.elsdDir;        
end
resPath = j2m(outDir,currentID);
if (~exist(resPath,'file'))
    resPath =strrep(resPath,'.mat','.txt');
    A = dlmread(resPath);
    [lines_,ellipses_] = parse_svg(A);
else
    load(resPath);
    lines_ = res.lineSegments;
    ellipses_ = res.ellipses;
end
%[