function [thin_contours] = hthin(contours, conex, outFile)
% Matlab interface for homotopic thinning by Michel Couprie:
% (http://www.esiee.fr/~coupriem/)
%
% syntax:
%   [thin_contours] = hthin(contours, conex,outFile)
%
% description:
%   Homotopic thinning of a boundary image
%
% arguments:
%   contours : uint8
%   outFile : (optional)
%   conex : output region connectivity (default 4)
%
% output:
%   thin_contours:   1-pixel wide boundaries
%
% note: uses writepgm from Matthew Dailey
%
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>
% July 2008

if nargin<3, outFile = 'tmp'; end
if nargin<2, conex = 4; end

tmpFile = strcat(outFile,'tmp.pgm');
writepgm(contours, tmpFile);
system(sprintf('util/hthin %s null %d -1 %s', tmpFile, conex, outFile));
thin_contours = imread(outFile);
delete(tmpFile);

if nargin<3, delete(outFile);end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function writepgm(image,filename)
%WRITEPGM Write a matrix as a raw pgm file
%         WRITEPGM(IMAGE,FILENAME) writes the array image as a raw PGM
%         file to FILENAME.
%
% Matthew Dailey 1999

if ((max(image(:))> 255) || (min(image(:))< 0)),
  error('Image pixels out of range.');
end;

% Open the file
%fprintf(1,'Opening %s...\n',filename);
fid = fopen(filename,'w');

if fid <= 0
  error('Could not open %s for writing.',filename);
end;

width = size(image,2);
height = size(image,1);

% Write header information
fprintf(fid,'P5\n');
fprintf(fid,'%d %d\n',width,height);
fprintf(fid,'255\n');

% Write the raw data -- transpose first though
count = fwrite(fid,image','uchar');

if count ~= width*height
  error('Could not write all data to %s', outfile);
end;

fclose(fid);


    
