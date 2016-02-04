function r = chopRight(r,p,isAbs)
if isAbs
    r(3) = r(3)-p;
else
    r(3) = ((1-p)*r(1)+p*r(3));
end