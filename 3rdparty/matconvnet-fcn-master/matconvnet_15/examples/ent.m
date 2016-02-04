function H = ent(P)
H = -sum(P.*log(P),1);