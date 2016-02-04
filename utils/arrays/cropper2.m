function [ I_cropped,Z ] = cropper2(I, rect, no_check)
%CROPPER cropper(I,rect) crop a rectangular region out of an image and pad
%by zeros if necessary (where rectangle is out of image boundaries)

if nargin < 3
    no_check = false;
end

if no_check
    Z = [];
    I_cropped = I(rect(2):rect(4),rect(1):rect(3),:);
    return;
end

rect = double(rect);
% rect = round(rect);
xmin = rect(1);
xmax = rect(3);
ymin = rect(2);
ymax = rect(4);

x_margin_left = max(0,-xmin+1);
y_margin_left = max(0,-ymin+1);
x_margin_right = max(xmax-size(I,2),1);
y_margin_right = max(ymax-size(I,1),1);

makeZ = nargout == 2;
if (makeZ)
    Z = true(size(I,1),size(I,2));
end

if (any([y_margin_left x_margin_left]))
    if (makeZ)
        Z = padarray(Z,[y_margin_left x_margin_left],0,'pre');
    end
    I = padarray(I,[y_margin_left x_margin_left],0,'pre');
end

if (any([y_margin_right x_margin_right]))
    if (makeZ)
        Z = padarray(Z,[y_margin_right x_margin_right],0,'post');
    end
    I = padarray(I,[y_margin_right x_margin_right],0,'post');
end

xmin = xmin+x_margin_left;
ymin = ymin+y_margin_left;
xmax = xmax+x_margin_left;
ymax = ymax+y_margin_left;

I_cropped = I(ymin:ymax,xmin:xmax,:);
if (makeZ)
    Z = Z(ymin:ymax,xmin:xmax);
end

end

