% Auto-generated by cameraCalibrator app on 02-Dec-2024
%-------------------------------------------------------


% Define images to process
imageFileNames = {'/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1303.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1302.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1301.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1300.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1259.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1258.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1256.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1255.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1253.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1252.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1251.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1250.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1249.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1248.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1247.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1246.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1245.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1244.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1241.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1240.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1239.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1238.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/2024-12-02_1235.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/camera_screenshot_02.12.2024.jpg',...
    '/MATLAB Drive/checker_board-20241202T040617Z-001/checker_board/camera_screenshot_02.12.2024_1.jpg',...
    };
% Detect calibration pattern in images
detector = vision.calibration.monocular.CheckerboardDetector();
minCornerMetric = 0.150000;
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames, 'MinCornerMetric', minCornerMetric);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates for the planar pattern keypoints
squareSize = 20.000000;  % in 밀리미터
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', '밀리미터', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

% See additional examples of how to use the calibration data.  At the prompt type:
% showdemo('MeasuringPlanarObjectsExample')
% showdemo('StructureFromMotionExample')
