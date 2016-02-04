function [gt_poly,angle] = loadGt(imgPath,gtDir)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
        [~,name,ext] = fileparts(imgPath);
        fName = fullfile(gtDir,[name ext  '.txt']);
        [objs,bbs] = bbGt( 'bbLoad', fName);
        [gt_poly,angle] = bbgt_to_poly(objs(1));

end

