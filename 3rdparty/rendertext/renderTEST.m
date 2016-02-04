in=imread('football.jpg');
out=rendertext(in,'OVERWRITE mode',[0 255 0], [1, 1]);
out=rendertext(out,'BLEND mode',[255 0 255], [30, 1], 'bnd', 'left');
out=rendertext(out,'left',[0 0 255], [101, 150], 'ovr', 'left');
out=rendertext(out,'mid',[0 0 255], [130, 150], 'ovr', 'mid');
out=rendertext(out,'right',[0 0 255], [160, 150], 'ovr', 'right');
imshow(out)