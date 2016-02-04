function bb = BoxUnion(bb)
bb = [min(bb(:,1:2),[],1) max(bb(:,3:4),[],1)];
end