function res=makeSaliencyMap(feaVec, pixelList, frameRecord, doNormalize, fill_value)

if (~iscell(pixelList))
    error('pixelList should be a cell');
end

if (nargin < 4)
    doNormalize = true;
end

if (nargin < 5)
    fill_value = 0;
end

h = frameRecord(1);
w = frameRecord(2);

top = frameRecord(3);
bot = frameRecord(4);
left = frameRecord(5);
right = frameRecord(6);

partialH = bot - top + 1;
partialW = right - left + 1;
res = CreateImageFromSPs(feaVec, pixelList, partialH, partialW, doNormalize);

if partialH ~= h || partialW ~= w
    feaImg = ones(h, w) * fill_value;
    feaImg(top:bot, left:right) = res;
    res = feaImg;
end