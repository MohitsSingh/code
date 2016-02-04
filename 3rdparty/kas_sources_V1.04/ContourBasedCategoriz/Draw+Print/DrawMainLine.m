function h = DrawMainLine(ml, col, thickness, draw_id, draw_dir, line_style)

% Adds main-line ml to current figure
%

% process arguments
if nargin < 3
  thickness = 1;
end
%
if nargin < 4
  draw_id = false;
end
%
if nargin < 5
  draw_dir = false;
end
%
if nargin < 6
  line_style = '-';
end


% Draw the main line
ef = [ml(3)+cos(ml(5))*ml(6)/2 ml(4)+sin(ml(5))*ml(6)/2];  % front endpoint
eb = [ml(3)-cos(ml(5))*ml(6)/2 ml(4)-sin(ml(5))*ml(6)/2];  % back  endpoint
h = plot([eb(1) ef(1)], [eb(2) ef(2)], 'Color', col, 'LineWidth', thickness, 'LineStyle', line_style);

% Draw indication of direction (front endpt)
if draw_dir
  h = [h; plot(ef(1), ef(2), '+b', 'LineWidth', thickness)];
  h = [h; plot(ef(1), ef(2), '+', 'Color', col, 'LineWidth', thickness)]; % b/w figs
end

% Draw id (some pixel away, in the direction perpendicular to the main-line)
if draw_id
  h = [h; text(ml(3)+ -2*sin(ml(5)), ml(4)+2*cos(ml(5)), num2str(ml(1)))];
end
