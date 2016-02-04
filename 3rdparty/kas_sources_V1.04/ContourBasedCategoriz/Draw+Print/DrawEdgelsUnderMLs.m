function DrawEdgelsUnderMLs(obj, mlids, col, thickness, draw_id)

% Draw edgels under main-lines mlids
%

global colors;

if nargin < 3
  col = colors(1,:);
end

if length(col) == 1                                         % an id has been given
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


hold on;
for m = mlids
  edgels = GetEdgelsUnderMLs(obj, m);
  
  %if not(even(m))                                          % draw adjancet segments in alternating colors
  %  plot(edgels(1,:), edgels(2,:), '--', 'Color', col, 'LineWidth', thickness);
  %else
    plot(edgels(1,:), edgels(2,:), '-', 'Color', col, 'LineWidth', thickness);
  %end
  if draw_id
    p = round(size(edgels,2)/2); midpt = edgels(:,p);
    o = obj.mainlines(5,m)+pi/2;                            % perp orient
    text(midpt(1)+cos(o)*6, midpt(2)+sin(o)*3, num2str(m), 'Color', col);
  end
  % Draw indication of direction
  % plot(edgels(1,end), edgels(2,end), '+b', 'LineWidth', thickness);
end
