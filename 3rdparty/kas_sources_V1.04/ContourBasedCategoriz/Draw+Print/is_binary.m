function b = is_binary(I)

% true if I is a binary image

b = max(max(I)) <= 1;
