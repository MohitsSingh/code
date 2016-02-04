function hh = quiver2(xy,uv,varargin)
    hh = quiver(xy(:,1),xy(:,2),uv(:,1),uv(:,2),varargin{:});