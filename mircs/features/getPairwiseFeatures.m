function f = getPairwiseFeatures(p,pN,g12)
% check the nature of the adjacency.
f = struct('flipStart',false,'flipEnd',false,'turnDirection',{});
f(1).flipStart = false;
switch g12
    case 1
        f.flipStart = true;
        p11 = p.endPoint; p12 = p.startPoint;
        p21 = pN.startPoint; p22 = pN.endPoint;
    case 2
        f.flipStart = true; f.flipEnd = true;
        p11 = p.endPoint; p12 = p.startPoint;
        p21 = pN.endPoint; p22 = pN.startPoint;
    case 3
        p11 = p.startPoint; p12 = p.endPoint;
        p21 = pN.startPoint; p22 = pN.endPoint;
    case 4
        f.flipEnd = true;
        p11 = p.startPoint; p12 = p.endPoint;
        p21 = pN.endPoint; p22 = pN.startPoint;
end

line_S = createLine(p11,p12);
line_T = createLine(p21,p22);
% get line-line features

turnDirection = 180*lineAngle(line_S,line_T)/pi;

f(1).turnDirection = turnDirection;
f(1).line_S = line_S;
f(1).line_T = line_T;
