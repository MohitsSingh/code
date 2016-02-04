function [regions,ovp,sel_] = chooseRegion(I,regions,minovp)
    clf;
    x2(I);
    bbox=getSingleRect(true);
%     h = imrect;
%     position = wait(h);
%     bbox = [position(1:2), position(3:4)+position(1:2)];
    
    
    [ovp,ints] = boxRegionOverlap2(bbox(1:4),regions);
    %regions = regions(ints./areas >= .3);
    sel_ = ovp >= .01;
    regions = regions(sel_);
    ovp = ovp(sel_);
    close(gcf);
end