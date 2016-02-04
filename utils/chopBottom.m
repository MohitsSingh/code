function r = chopBottom(r,p,isAbs)
if isAbs
    r(4) = r(4)-p;
else    
    r(4) = ((1-p)*r(2)+p*r(4));
end