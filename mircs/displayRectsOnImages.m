function [rects] = displayRectsOnImages(rects,images)


[ Z,Zind,x,y ] = multiImage(images,false);
figure,imshow(Z)
for k = 1:length(images)
    rects(k,1:4) = rects(k,1:4)+[x(k) y(k) x(k) y(k)];
end
hold on;
plotBoxes2(rects(:,[2 1 4 3]),'color','green','LineWidth',2);

