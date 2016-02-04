function r = chopTop(r,p,isAbs)
if isAbs
    r(2) = r(2)+p;
else
    r(2) = (p*r(2)+(1-p)*r(4));
end