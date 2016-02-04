function I = showRegionPair(A,B)
int = A & B;
A_BC = A & ~B;
AC_B = B & ~A;
I = im2uint8(cat(3,int,A_BC,AC_B));
end