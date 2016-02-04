function R = cropToNonzero(R)
    [yy,xx] = find(R);
    R = cropper(R,pts2Box([xx yy]));
end