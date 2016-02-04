function bb = region2Box(r)
[yy xx] = find(r);
bb = pts2Box([xx yy]);
end