function xy1xy2 = boundrect(XY,expand)
%BOUNDRECT Bounding rectangle of XY points.
%  xy1xy2 = boundrect(XY,expand)
%  expand = nonnegative expansion factor for bounding rectangle, where
%           expansion of "expand" times max X,Y extent is added to all sides
%           of rectangle
%         = 0 (default)
%  xy1xy2 = bounding rectangle
%         = [min(X) min(Y); max(X) max(Y)]

% Copyright (c) 1994-2002 by Michael G. Kay
% Matlog Version 6 19-Sep-2002

% Input Error Checking ******************************************************
if nargin < 2 | isempty(expand), expand = 0; end

if isempty(XY) | size(XY,2) ~= 2 | ~isnumeric(XY)
   error('XY not a valid two-column matrix.')
elseif ~isempty(expand) & ...
      (length(expand(:)) ~= 1 | ~isfinite(expand) | expand < 0)
   error('"expand" must be a nonnegative scalar.')
end
% End (Input Error Checking) ************************************************

xy1xy2 = [min(XY,[],1); max(XY,[],1)];
offset = max(diff(xy1xy2)) * expand;
if offset > 0
   xy1xy2 = [xy1xy2(1,:) - offset; xy1xy2(2,:) + offset];
end
