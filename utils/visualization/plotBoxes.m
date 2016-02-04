function plotBoxes(boxes,varargin) % xmin ymin xmax ymax
if (nargin == 1)
    varargin = {'g-','LineWidth',2};
end
if (~isempty(boxes))
    plotBoxes2(boxes(:,[2 1 4 3]),varargin{:});
end



end