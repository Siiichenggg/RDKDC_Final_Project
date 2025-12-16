function mu = manipulability(J, measure)
    % Input:
    %   J: 6x6 Jacobian matrix
    %   measure: string specifying the manipulability measure
    %            - 'sigmamin': minimum singular value
    %            - 'detjac': determinant of Jacobian
    %            - 'invcond': inverse condition number
    %
    % Output:
    %   mu: scalar manipulability measure


    if strcmp(measure, 'sigmamin')
        % Minimum singular value of J
        sigma = svd(J);
        mu = min(sigma);

    elseif strcmp(measure, 'detjac')
        mu = abs(det(J));

    elseif strcmp(measure, 'invcond')
        % Value between 0 and 1, and 1 is best 
        sigma = svd(J);
        sigma_min = min(sigma);
        sigma_max = max(sigma);
        mu = sigma_min / sigma_max;

    else
        error('Invalid measure. Must be ''sigmamin'', ''detjac'', or ''invcond''');
    end
end
