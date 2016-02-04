function I = maskedPatch(I,curMask,toCrop,fillValue,tight)
if nargin < 3
    toCrop = true;
end
if nargin < 4 || isempty(fillValue)
    fillValue = .5;
    if isa(I,'uint8')
        fillValue = 128;
    end
end

if nargin < 5
    tight = false;
end

m = repmat(~curMask,[1 1 3]);
if ~tight
    m = imerode(m,ones(15));
end
I(m) = fillValue;
if toCrop
    curMaskBox = round(makeSquare(region2Box(curMask),true));
    I = cropper(I,curMaskBox);
end