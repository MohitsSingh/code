function ecs = DetectEdgelChains(I)

% Detects all edgel chains in binary image I,
% and returns points in chains ecs(i).chain.
%
% The chains must be disjoint, as no junctions are allowed.
% Each chain which might be open or closed.
%

ecs = [];
needmore = true;
while needmore
   % Detect a chain ec
   ec = DetectEdgelChain(I);

   % Add ec to output
   ecs = [ecs ec];

   % Remove ec from I
   for i = 1:size(ec.chain,2)
       p = ec.chain(:,i);
       %I(p(2)+1,p(1)+1) = w;   % '+1' -> pixel coords to matrix coords
       I((p(2)-1):(p(2)+3),p(1)-1) = 1;       % to remove 'double-lines' and 4-connected parts
       I((p(2)-1):(p(2)+3),p(1)) = 1; 
       I((p(2)-1):(p(2)+3),p(1)+1) = 1; 
       I((p(2)-1):(p(2)+3),p(1)+2) = 1;     
       I((p(2)-1):(p(2)+3),p(1)+3) = 1; 
   end

   needmore = (not(isempty(find(I==0))));
end
