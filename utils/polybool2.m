function res = polybool2(operation,p1,p2,varargin)
     [x3, y3] = polybool(operation, p1(:,1), p1(:,2), p2(:,1), p2(:,2), varargin{:});
     res = [x3 y3];
end