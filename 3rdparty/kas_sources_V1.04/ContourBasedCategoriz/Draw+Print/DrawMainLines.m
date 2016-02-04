function DrawMainLines(mls, col, ids, thickness, draw_id, draw_dir)

% Draws main-lines mls over the current figure
%
% Input:
% mls(:,i) are 6x1 vectors in the form:
%
% [id chain x y orientation length]'
%
% with orientation in [0,2pi]
%

global colors;

if nargin < 2
  col = colors(1,:);
end

if length(col) == 1  % an id has been given
  col_id = col;
  col_id = mod(col_id-1,size(colors,1))+1;
  col = colors(col_id,:);
end

if nargin < 4
  thickness = 1;
end

if nargin < 5
  draw_id = false;
end

if nargin < 6
  draw_dir = false;
end

if nargin > 2 & not(ischar(ids))
  % selects main-lines with input ids
  [ids trash mlix] = intersect(ids, mls(1,:));
  mls = mls(:,mlix);
end

hold on;
for mlix = 1:size(mls,2)
   DrawMainLine(mls(:,mlix), col, thickness, draw_id, draw_dir);
end
hold off;
