function mouth_poly = fix_mouth_poly(mouth_poly)

if (size(mouth_poly,1)==13)
    p1 = mouth_poly(13,:);p2 = mouth_poly(7,:);    
else
    p1 = mouth_poly(1,:);p2 = mouth_poly(4,:);    
end


V = p2-p1;
V_theta = atan2(V(2),V(1));

mouth_width = norm(p1-p2);
mouth_center = (p1+p2)/2;
%plotEllipse2(cRow,cCol,ra,rb,phi,color,nPnts,lw,ls, show)
[~,x,y] = plotEllipse2(mouth_center(2),mouth_center(1),mouth_width/2,mouth_width/8,V_theta,'b',20,1,'-',false);
mouth_poly = [x(:) y(:)];