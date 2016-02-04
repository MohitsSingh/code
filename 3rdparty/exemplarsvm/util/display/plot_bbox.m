function plot_bbox(bb,titler,col1,col2,do_spacing,linewidths,hg_size)
%Plot a bounding box in the image with an optional title,
%inner/outer colors, and a boolean flag indicating whether the box
%should be dotted or not

%Tomasz Malisiewicz (tomasz@cmu.edu)
if numel(bb)==0
  return;
end
%Needs to be a double or text command complains (thanks Carl)
bb = double(bb);

if ~exist('do_spacing','var')
  do_spacing = 0;
end

if ~exist('col2','var')
  col2 = [1 0 0];
end
if ~exist('col1','var')
  col1 = [0 0 1];
end

if ~exist('titler','var')
  titler = '';
end

if isnumeric(titler)
  titler = num2str(titler);
elseif ~isstr(titler)
  titler = 'invalid';
end

if ~exist('linewidths','var')
  linewidth1 = 3;
  linewidth2 = 1;
  linewidths = [3 1];
else
  linewidth1 = linewidths(1);
  linewidth2 = linewidths(2);
end

if ~exist('hg_size','var')
  hg_size = [1 1];
end

if prod(hg_size) > 1
  %%% draw grids here
  x1 = bb(:,3) - bb(:,1);
  y1 = bb(:,4) - bb(:,2);
  
  for i = 1:(hg_size(2))
    for j = 1:(hg_size(1))
      curbb = bb;
      curbb(:,1) = curbb(:,1) + (i-1)*x1/hg_size(2);
      curbb(:,3) = curbb(:,1) + (1)*x1/hg_size(2);
            
      curbb(:,2) = curbb(:,2) + (j-1)*y1/hg_size(1);
      curbb(:,4) = curbb(:,2) + (1)*y1/hg_size(1);

      plot_bbox(curbb,titler,col1,col2,do_spacing,[1 1]);%linewidths);
      
    end
  end
end

if size(bb,1) > 1
  for i = 1:size(bb,1)
    plot_bbox(bb(i,:),titler,col1,col2,do_spacing,linewidths);
    hold on;
  end
  return;
end

if do_spacing == 1
  
  order1 = [1 3 3 1 1 3];
  order2 = [2 2 4 4 2 2];
  for i = 1:length(order1)-1
    bx = linspace(bb(order1(i)),bb(order1(i+1)),10);
    by = linspace(bb(order2(i)),bb(order2(i+1)),10);
    hold on;
    plot(bx,by,'--','Color',col1,'linewidth',linewidth1);
    hold on;
    plot(bx,by,'--','Color',col2,'linewidth',linewidth2);
  end
else
  hold on;
  plot(bb([1 3 3 1 1 3]),bb([2 2 4 4 2 2]),'Color',col1,'linewidth',linewidth1);
  hold on;
  plot(bb([1 3 3 1 1 3]),bb([2 2 4 4 2 2]),'Color',col2,'linewidth',linewidth2);
end

fontscale = (bb(3)-bb(1)+1)*(bb(4)-bb(2)+1) / (100*100);

if exist('titler','var') & length(titler)>0
  hold on;
  text(bb(1),bb(2),titler,'color','k','backgroundcolor',col1,...
       'verticalalignment','bottom','horizontalalignment','left', ...
       'fontsize',min(50,max(10,round(fontscale))));
end
