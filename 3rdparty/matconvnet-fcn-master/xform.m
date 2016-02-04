function I = xform(I,rot,scale,orig_sz)
I = imrotate(I,rot,'bilinear','crop');
I = imResample(I,scale,'bilinear');
padSize = (orig_sz-size(I,1))/2;
if padSize > 0
    I = padarray(I,[1 1]*floor(padSize),'pre');
    I = padarray(I,[1 1]*ceil(padSize),'post');
elseif padSize < 0
    padSize = -padSize;
    I = I(ceil(padSize):ceil(padSize)+orig_sz-1,ceil(padSize):ceil(padSize)+orig_sz-1);
end