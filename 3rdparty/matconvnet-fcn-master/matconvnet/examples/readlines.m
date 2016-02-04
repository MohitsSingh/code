function [ lines ] = readlines( filename )
%READLINES Summary of this function goes here
%   Detailed explanation goes here
lines = {};
fid = fopen(filename);
line = fgetl(fid);
while (line~=-1)
    lines{end+1} = line;
    line = fgetl(fid);
end

end

