function ints = BoxIntersection2(boxes1,boxes2)
    n1 = size(boxes1,1);
    n2 = size(boxes2,1);
    ints = zeros(n1,n2);
    for k = 1:n1
        bb = BoxIntersection(boxes1(k,:),boxes2);
        [~,~,s] = BoxSize(bb);
        ints(k,:) = s;
    end
end