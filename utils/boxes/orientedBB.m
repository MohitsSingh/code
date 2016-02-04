function p = orientedBB(p0,v1,width)
    v1 = row(v1);
    v = v1/norm(v1);
    v2 = [-v(2) v(1)];
    v2 = v2*width;
    p = [p0;...
        p0 + v1;...
        p0 + v1+v2;...
        p0 + v2];    
end
