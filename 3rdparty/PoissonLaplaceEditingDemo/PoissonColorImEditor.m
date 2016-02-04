function TargFilled = PoissonColorImEditor(TargImPaste, MaskTarg)
%Use: Split color image into 3 colors and paste source into target image
%seperately
%Input:
%   TargImPaste - Target image with Laplacian of source image within the
%   masked area
%   MaskTarg    - Mask area for source to be pasted into
%Output:
%   TargFilled  - Target image with masked area solved by Poisson editor

TargImPasteR = TargImPaste(:, :, 1);
TargImPasteG = TargImPaste(:, :, 2);
TargImPasteB = TargImPaste(:, :, 3);

AdjacencyMat = calcAdjancency( MaskTarg );
TargBoundry  = bwboundaries( MaskTarg, 8);

TargFilledR = PoissonGrayImEditor(TargImPasteR, MaskTarg, AdjacencyMat, TargBoundry);
TargFilledG = PoissonGrayImEditor(TargImPasteG, MaskTarg, AdjacencyMat, TargBoundry);
TargFilledB = PoissonGrayImEditor(TargImPasteB, MaskTarg, AdjacencyMat, TargBoundry);

TargFilled = cat(3, TargFilledR, TargFilledG, TargFilledB);