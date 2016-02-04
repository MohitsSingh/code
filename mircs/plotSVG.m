function plotSVG(A,offset)
if (nargin < 2)
    offset = [0 0];
end


for k = 1:size(A,1)
    %     k=13
    %
    a = A(k,:);
    thetaStart = a(6)-a(5);
    thetaEnd = a(7)-a(5);
    if (all(a(5:7)==-1)) % line
%         k
        plot(a([1 3])-offset(1)+1,a([2 4])-offset(2)+1,'r','LineWidth',2);
    else
        plotEllipse2(a(2)-offset(2)+1,a(1)-offset(1)+1,a(3),a(4),[a(5) thetaStart thetaEnd],'g',100,2);
    end
    
    %     pause
end
end