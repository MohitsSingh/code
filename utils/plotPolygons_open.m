
function plotPolygons_open(polys,varargin)
hold on;if (isempty(polys)),return;end
if (~iscell(polys))
    polys = {polys};
end
polys = polys(:);
for k = 1:size(polys,1)
    if isempty(polys{k})
        continue
    end
    x = polys{k}(:,1); y = polys{k}(:,2);
    plot(x,y,varargin{:});
end