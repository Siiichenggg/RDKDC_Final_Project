function dirs = rtde_path_setup()
%RTDE_PATH_SETUP Add required paths and ensure the correct ur_rtde_interface.m is used.

thisDir = fileparts(mfilename('fullpath'));

% Put RTDE folder first so RTDE/ur_rtde_interface.m wins over RTDE/Matlab/ur_rtde_interface.m.
addpath(thisDir, '-begin');

matlabDir = fullfile(thisDir, 'Matlab');
helperDir = fullfile(matlabDir, 'helper_function');

% Add kinematics helpers later in the path to avoid class shadowing.
if exist(helperDir, 'dir')
    addpath(helperDir, '-end');
end
if exist(matlabDir, 'dir')
    addpath(matlabDir, '-end');
end

% Verify the interface class resolves to RTDE/ur_rtde_interface.m.
resolved = which('ur_rtde_interface');
expected = fullfile(thisDir, 'ur_rtde_interface.m');
if isempty(resolved) || isempty(strfind(resolved, expected))
    error(['Path conflict: expected ur_rtde_interface at:\n  %s\n' ...
           'but MATLAB resolves it to:\n  %s\n' ...
           'Fix by ensuring RTDE/ is ahead of RTDE/Matlab/ on the MATLAB path.'], ...
          expected, resolved);
end

dirs = struct('thisDir', thisDir, 'matlabDir', matlabDir, 'helperDir', helperDir);
end

