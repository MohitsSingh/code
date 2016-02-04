function Z = make_disparity(I,src,dst)
    
    [X,Y]=meshgrid(1:size(I,2),1:size(I,1));
    n = -src(:,1)+dst(:,1);
    %n = sum((src-dst).^2,2).^.5;
    Z = griddata(src(:,1),src(:,2),n,X,Y,'nearest');
    Z(Z<0) = 0;Z(isnan(Z)) = 0;
end