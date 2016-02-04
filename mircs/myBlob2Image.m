function B = myBlob2Image(blob,I)
% return a binary mask in image coordinates of this blob.
B = false(size2(I));
B(blob.rect(1):blob.rect(3),blob.rect(2):blob.rect(4)) = blob.mask;
end