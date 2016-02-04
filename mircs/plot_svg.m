function plot_svg(lines_,ellipses_,d)
if (nargin < 3)
    d = false(size(ellipses_,1),1);
end
for iLine = 1:size(lines_,1)
    a = lines_(iLine,:);
    plot(a([1 3]),a([2 4]),'g','LineWidth',1);%,'LineSmoothing','on');
end
o1 = [5 6 7];
for iEllipse = 1:size(ellipses_,1)
    %         iEllipse
    a = ellipses_(iEllipse,:);
    if (a(3)==a(4)) %circle
        plotEllipse2(a(1),a(2),a(3),a(4),a(o1),'b',20,1);
    else
        plotEllipse2(a(1),a(2),a(3),a(4),a(o1),'r',20,1);
    end
    
    if (d(iEllipse))
        plotEllipse2(a(1),a(2),a(3),a(4),a(o1),'m',20,2);
    end
    %         if (d1(iEllipse))
    %             plotEllipse2(a(1),a(2),a(3),a(4),a(5:7),'k',20,2);
    %         end
    %         pause;
end