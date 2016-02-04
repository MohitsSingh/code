function DrawLinks(ecs, links, draw_id)

% Draws links among pairs of edgelchains in ecs
% 
% Input:
% links(:,l) = [c1; c2; endpt1; endpt2; link_pt]
%            = chain ecs(c1) links to ecs(c2) at link_pt, near endpoint endpt1;
%              if endpt2>0 then endpoint endpt2 of ecs(c2) is near link_pt
%

if nargin < 3
  draw_id = false;
end

DrawEdgelChains(ecs);
hold on;
n = 0;
for l = links
   % Build search area
   [A trash] = BuildTrapeze(ecs(l(1)), l(3));

   % Draw search area
   if l(3) == 1 col = 'g'; else col = 'r'; end

   %col = 'k';  % b/w
   %plot([A(1,:) A(1,1)], [A(2,:) A(2,1)], 'color', [0.5 0.5 0.5], 'LineWidth', 3);  % paper figs
   %plot([A(1,:) A(1,1)], [A(2,:) A(2,1)], col, 'LineWidth', 6); % paper figs
   plot([A(1,:) A(1,1)], [A(2,:) A(2,1)], col, 'LineWidth', 2);

   % Draw linking point
   %plot(ecs(l(2)).chain(1,l(5)), ecs(l(2)).chain(2,l(5)), ['o' col], 'LineWidth', 6);  % paper figs
   plot(ecs(l(2)).chain(1,l(5)), ecs(l(2)).chain(2,l(5)), ['o' col], 'LineWidth', 2);

   % Draw id
   n=n+1;
   if draw_id
     text(ecs(l(2)).chain(1,l(5))+4, ecs(l(2)).chain(2,l(5)), num2str(n), 'color', [1 0 0], 'FontSize', 10);
   end
end
