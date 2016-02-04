 function r = chopLeft(r,p,isAbs)
 if isAbs
     r(1) = r(1)+p;
 else
    r(1) = (p*r(1)+(1-p)*r(3));
 end