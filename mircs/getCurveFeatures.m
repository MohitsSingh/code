function curveFeatures = getCurveFeatures(conf,curImageData)

[lines_,ellipses_] = getELSDResults(conf,curImageData.imageID);
global pTypes;
pTypes.TYPE_LINE = 1;
pTypes.TYPE_ELLIPSE = 2;
pTypes.TYPE_POINT = 3;
pTypes.TYPE_POLYGON = 4;
pTypes.special.face = 5000;
pTypes.special.mouth = 50001;
ellipseFeats = getEllipseFeatures(ellipses_);
lineFeats = getLineFeatures(lines_);
faceBox = inflatebbox(curImageData.faceBox,3*[1.5 1.5],'both',false);
[I,I_rect] = getImage(conf,curImageData.imageID);
faceBox = faceBox+I_rect([1 2 1 2]);
curveFeatures.lineFeats = removeByBB(lineFeats(:),faceBox);
curveFeatures.ellipseFeats = removeByBB(ellipseFeats(:),faceBox);

figure,imshow(I); hold on; plot_svg(cat(1,curveFeatures.lineFeats.params),cat(1,curveFeatures.ellipseFeats.params))

function L = removeByBB(L,b)
s = cat(1,L.bbox);
ints = BoxSize(BoxIntersection(s,b));
L = L(ints > 0);

