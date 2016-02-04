function bb = shiftBoxes(bb,x,y)
bb(:,1:4) = bb(:,1:4) + [x y x y];
end