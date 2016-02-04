function ec = DetectEdgelChain(I)

% Detects an edgel chain
% in binary image I, and returns points
% in chain ec.
% Toy routine to well test the
% subsequent step: determining the types
% and attributes of the fragments on the chain
% (via spline toolbox).
% If there is more than one chain in I, only the top-left one is extracted.
% The chain which might be open or closed (no junctions allowed!)
%

% Find any point (0 = figure, 1 = ground)
w = size(I,2);
h = size(I,1);
t = find(I==0);
cp(1) = rem(t(1),h);
cp(2) = (t(1)-cp(1))/h+1;   % current pixel
I(cp(1), cp(2)) = 1;        % delete pixel
cp = cp';
cont = cp;                  % current contour

% 8 directions, counterclockwise ordered
D = [1 1 0 -1 -1 -1  0  1;
     0 1 1  1  0 -1 -1 -1];

% Follow contour counterclockwise starting from cp
for k = 1:2

needmore = true;
cd = 1;      % current direction
while needmore
   needmore = false;
 
   % find next contour point
   for nd = 1:8
     dv = D(:,cd);   % direction vector
     if I(cp(1)+dv(1), cp(2)+dv(2)) == 0
        cp = cp + dv;
        cont = [cont cp];
        needmore = true;
        I(cp(1),cp(2)) = 1;
        cd = cd - 2;   % works also if 'interior' present (to some degree)
        if cd < 1
          cd = cd+8;
        end
        break;
     else
        cd = cd + 1;
        if cd > 8
           cd = cd-8;
        end
     end
   end  % try next direction 
end

% Follow contour clockwise starting from cont(:,1)
D = reverse(D);
if length(cont) > 0    % happens in second iter (k=2) for closed curves or when starting from an endpoint
  cp = cont(:,1);
end
cont_tot(k).cont = cont;
cont = [];

end

% assemble contour
cont = [reverse(cont_tot(1).cont) cont_tot(2).cont];

% Ajdust contour coords to pixel coords
cont = [cont(2,:)-1; cont(1,:)-1];

% Format return chain
ec.chain = cont;
