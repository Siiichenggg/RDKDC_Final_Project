function Ad = adjoint(g)
    R = g(1:3, 1:3);
    p = g(1:3, 4);

    % Compute [p]R using SKEW3
    p_skew = SKEW3(p);
    p_skew_R = p_skew * R;

    % Construct the 6x6 Adjoint matrix
    % Ad = [R,      [p]R  ]
    %      [0_{3x3},  R   ]
    Ad = [R,           p_skew_R;
          zeros(3,3),  R        ];
end
