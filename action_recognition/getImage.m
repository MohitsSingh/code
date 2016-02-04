function [ I ] = getImage( conf,I )
%GETIMAGE Summary of this function goes here
%   Detailed explanation goes here

if (exist(I,'file'))
    I = imread2(I);
else
    I = imread2(fullfile(conf.baseDir,I));
end

end

