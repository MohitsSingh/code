function a = polyarea2(xy)
    if isempty(xy)
        a = 0;
        return;
    end
    a = polyarea(xy(:,1),xy(:,2));
end