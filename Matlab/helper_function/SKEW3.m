function S=SKEW3(v)
      x1 = v(1); x2 = v(2); x3 = v(3);
      S =  [ 0, -x3,  x2;
            x3,   0, -x1;
           -x2,  x1,   0 ];
end