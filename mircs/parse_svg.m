function [lines,ellipses] = parse_svg(A,offset)
if (nargin < 2)
    offset = [0 0];
end
lines = {};
ellipses = {};
for k = 1:size(A,1)
    %     k=13
    %
    a = A(k,:);
    thetaStart = a(6)-a(5);
    thetaEnd = a(7)-a(5);
    
    if (all(a(5:7)==-1)) % line
        lines{end+1} = a(1:4)-offset([1 2 1 2])+2;
    else
        ellipses{end+1} = [a(2)-offset(2)+2,a(1)-offset(1)+2,a(3:5) thetaStart thetaEnd];
    end
    
    %     pause
end

lines = cat(1,lines{:});
ellipses = cat(1,ellipses{:});
end