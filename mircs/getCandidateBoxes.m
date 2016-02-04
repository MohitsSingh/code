function bb = getCandidateBoxes(p,props)
debug_ = false;
a = 0;
bb = cat(1,props.BoundingBox);
bb(:,[3 4]) = bb(:,[3 4])+bb(:,[1 2]);
%     plotBoxes2(bb(:,[2 1 4 3]));
%  get the face box..
%         curFaceBox = imageDataStruct.faceBoxes(k,:);
%         plotBoxes2(p(:,[2 1 4 3]),'g-.','LineWidth',2);
expectedCupBox = [p(1)-10,...
    (p(2)+p(4))/2,...
    p(3),...
    2*p(4)-p(2)];
%     expectedCupBox = inflatebbox(expectedCupBox,[1 2],'post');
expectedCupBox = inflatebbox(expectedCupBox,[2 1],'both');
if (debug_)
    plotBoxes2(expectedCupBox(:,[2 1 4 3]),'g-.','LineWidth',2);
end

ints_ = BoxIntersection(expectedCupBox,bb);
uns_ = BoxUnion(expectedCupBox,bb);
[a_ b_ areas] = BoxSize(bb);
[a_ b_ ints_] = BoxSize(ints_);
[a_ b_ uns_] = BoxSize(uns_);

bb = bb((ints_./areas)>.7,:);


end