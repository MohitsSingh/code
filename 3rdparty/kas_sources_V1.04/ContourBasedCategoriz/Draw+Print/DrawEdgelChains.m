function DrawEdgelChains(ecs, new_fig, draw_id, draw_cv, bw, Ts)

% Draws edgel chains ecs(i).chain
%
% if bw = true -> draw in black/white scheme
% if bw = 1x3 color vector -> draw in color bw
% if Ts = [T' s ctr'] given -> translate the ecs by T, scale it by s, wrt the transl center ctr
%
% Ts is optional
%

global colors;

% process arguments
if nargin < 2
   new_fig = true;
end
if nargin < 3
   draw_id = false;
end
if nargin < 4
   draw_cv = false;
end
if nargin < 5
   bw = false;
end
if nargin < 6
   Ts = false;
end

if new_fig
  iptsetpref('ImshowBorder','tight');  % slides
  figure;
  axis ij; hold on;             % comment out to avoid border
  %imshow(ones(464,498)*255);   % starbucks
  %imshow(ones(375,500)*255);   % brookfield
  %imshow(ones(400,500)*255);   % stilllife
  %imshow(ones(540,800)*255);   % car (applelogo)
  %imshow(ones(300,200)*255);   % mybottle9
  %imshow(ones(300,200)*255);   % myapplelogo4
  %imshow(ones(404,600)*255);   % '121' (jurie horses)
  %imshow(ones(395,593)*255);   % '311' (jurie horses)
  %imshow(ones(375,500)*255);   % sarah, work, nero, hockey, wake (mugs)
  %imshow(ones(500,393)*255);   % reusable (mugs)
  %imshow(ones(253,275)*255);   % ridgid (mugs)
  %imshow(ones(398,509)*255);   % tall (mugs)
  %imshow(ones(253,167)*255);   % witch (mugs)
  %imshow(ones(372,494)*255);   % clutter (mugs)
  %imshow(ones(498,372)*255);   % kids (mugs)
  %imshow(ones(272,300)*255);   % relty (mugs)
  axis equal;                   % for slides, notice it can cause border effects for images that are taller than wider
else
  axis ij; axis equal;
end

hold on;
for i = 1:length(ecs)
  ec = ecs(i).chain;

  if length(Ts) > 1
    ec = TrafoPoints(ec, Ts(1:2)', Ts(3), Ts(4:5)');
  end

  if length(bw) == 1 & not(bw)
     c = rem((i-1),size(colors,1))+1;
     %plot(ec(1,:), ec(2,:), 'Color', colors(c,:));
     plot(ec(1,:), ec(2,:), 'Color', colors(c,:), 'LineWidth', 2);       % for slides
     %plot(ec(1,:), ec(2,:), 'Color', [1 0 0], 'LineWidth', 2);           % for slides
  elseif length(bw) == 1 & bw      % just draw in stdr bw color scheme
     %plot(ec(1,:), ec(2,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 1);     % stdr working
     %plot(ec(1,:), ec(2,:), 'Color', [0.5 0.5 0.5], 'LineWidth', 1);     % bmvc06 + pami + cvpr07 papers
     %plot(ec(1,:), ec(2,:), 'Color', [0.01 0 0], 'LineWidth', 1);         % cvpr07 paper + slides
     %plot(ec(1,:), ec(2,:), 'Color', [0.99 1 1], 'LineWidth', 3);  
     plot(ec(1,:), ec(2,:), 'Color', [0.3 0.3 0.3], 'LineWidth', 2);      % cvpr07 slides
  elseif length(bw) == 3     % bw is actually a desired color
     plot(ec(1,:), ec(2,:), 'Color', bw, 'LineWidth', 1);  
  else
     error('DrawEdgelsChains: bw must be either true/false, or a 1x3 color vector');
  end
  %plot(ec(1,:), ec(2,:), 'Color', colors(c,:), 'LineWidth', 4);  % for slides
  %plot(ec(1,:), ec(2,:), '.', 'Color', [0 0 0], 'LineWidth', 2);       % for slides
  %plot(ec(1,:), ec(2,:), 'Color', [0 0 0], 'LineWidth', 1);       % for paper figs
  %plot(ec(1,:), ec(2,:), 'color', [0.5 0.6 0.5], 'LineWidth', 6);  % for paper figs
  %plot(ec(1,:), ec(2,:), 'color', [0.6 0.78 0.6], 'LineWidth', 12);   % for paper figs
  %plot(ec(1,:), ec(2,:), 's', 'color', [0.75 0.82 0.75], 'LineWidth', 3);  % for paper figs

  if draw_id && not(Ts)              % if transl+scale draw_id not supported
    p = ceil(size(ec,2)/2);
    text(ec(1,p), ec(2,p), num2str(i), 'Color', colors(c,:));
  end
  
  if draw_cv && not(Ts)              % if transl+scale draw_cv not supported
    %fnplt(ecs(i).cv,'Color',colors(c,:));                 
    fnplt(ecs(i).cv,'g');  
  end
end
