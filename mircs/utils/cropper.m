function [ I_cropped,Z ] = cropper(I, rect)
%CROPPER cropper(I,rect) crop a rectangular region out of an image and pad
%by zeros if necessary (where rectangle is out of image boundaries)

% rect = round(rect);
xmin = rect(1);
xmax = rect(3);
ymin = rect(2);
ymax = rect(4);

x_margin_left = max(0,-xmin+1);
y_margin_left = max(0,-ymin+1);
x_margin_right = max(xmax-size(I,2),1);
y_margin_right = max(ymax-size(I,1),1);
Z = true(size(I,1),size(I,2));

Z = padarray(Z,[y_margin_left x_margin_left],0,'pre');
Z = padarray(Z,[y_margin_right x_margin_right],0,'post');
I = padarray(I,[y_margin_left x_margin_left],0,'pre');
I = padarray(I,[y_margin_right x_margin_right],0,'post');

xmin = xmin+x_margin_left;
ymin = ymin+y_margin_left;
xmax = xmax+x_margin_left;
ymax = ymax+y_margin_left;

I_cropped = I(ymin:ymax,xmin:xmax,:);
Z = Z(ymin:ymax,xmin:xmax);

end

