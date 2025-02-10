clc; clear all; close all;

%%
tic
% Load the main data file interactively
[file, path] = uigetfile('*.mat', 'Select the main data MAT file');
if isequal(file, 0)
    disp('User cancelled file selection.');
    return;
end
fullFileName = fullfile(path, file);
data = load(fullFileName);

% Required fields check
reqFields = {'x','y','velocity_magnitude','u_component','v_component','v_original'};
for k = 1:length(reqFields)
    if ~isfield(data, reqFields{k})
        error('Missing required variable: %s', reqFields{k});
    end
end
toc
%%
% Load a mask file interactively.
% In the mask file, the variable "mask" should be logical.
% Option A: walls are 1 and ROI is 0 (default below).
% Option B: if your mask is reversed (walls are 0, ROI is 1),
%           then invert the mask by uncommenting the indicated line.

useMaskFile = true;
if useMaskFile
    [maskFile, maskPath] = uigetfile('*.mat', 'Select the MASK file (must contain variable ''mask'')');
    if isequal(maskFile, 0)
        disp('User cancelled mask file selection. Using v_original for wall detection.');
        useMaskFile = false;
    else
        fullMaskFile = fullfile(maskPath, maskFile);
        maskData = load(fullMaskFile);
        if ~isfield(maskData, 'mask')
            error('Mask file does not contain a variable named ''mask''.');
        end
        userMask = logical(maskData.mask);

        % Option A: if your mask is defined such that walls are marked as 1:
        wall = userMask;

        % Option B: if your mask is defined such that walls are marked as 0, and ROI is 1,
        % then uncomment the following line to invert the mask:
        % wall = ~userMask;
    end
end

% If not using the mask file, detect walls from v_original (locations that are NaN in all frames)
if ~useMaskFile
    numFrames = numel(data.v_original);
    wall = true(size(data.v_original{1}));
    for k = 1:numFrames
        wall = wall & isnan(data.v_original{k});
    end
    wall = squeeze(wall);
end

%%
% AVERAGE VELOCITY MAGNITUDE
velStack = cat(3, data.velocity_magnitude{:});
avgVel   = squeeze(nanmean(velStack, 3));

% Resize wall mask to match avgVel dimensions if necessary
if ~isequal(size(avgVel), size(wall))
    warning('Resizing wall mask from %s to %s', mat2str(size(wall)), mat2str(size(avgVel)));
    wall = imresize(wall, size(avgVel), 'nearest');
end

% Set wall areas to NaN in avgVel
avgVel(wall) = NaN;

%
% TIME-AVERAGE THE U & V COMPONENTS
uStack = cat(3, data.u_component{:});
vStack = cat(3, data.v_component{:});

avgU = squeeze(nanmean(uStack, 3));
avgV = squeeze(nanmean(vStack, 3));

% Apply the wall mask to avgU and avgV as well
avgU(wall) = NaN;
avgV(wall) = NaN;

%
% PLOT: contourf + quiver
X = data.x{1};
Y = data.y{1};

figure('Position',[100 100 900 600]); 
hold on;

% 1) Filled contour of the avg velocity magnitude
contourf(X*1000, Y*1000, avgVel, 20, 'LineColor', 'none');

% 2) Overplot the wall (white mask)
if any(wall(:))
    contourf(X*1000, Y*1000, wall, [0.5 0.5], 'FaceColor', 'w', 'LineStyle', 'none');
end

% 3) Quiver: skip points for clarity (e.g., every 5th point)
skip = 5;
% scaling = 1;
quiver(...
    X(1:skip:end,1:skip:end)*1000, ...
    Y(1:skip:end,1:skip:end)*1000, ...
    avgU(1:skip:end,1:skip:end), ...
    avgV(1:skip:end,1:skip:end), ...
    'k' ...    % color of arrows
);

% Adjust colormap and color axis
cmap = parula;
cmap(end+1, :) = 1;  % Append white for NaNs
colormap(cmap);
caxis([min(avgVel(~isnan(avgVel))), max(avgVel(~isnan(avgVel)))]);

colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Average Velocity Magnitude');
axis equal;
hold off;


%%

%
figure; 
contourf(X*1000, Y*1000, avgVel, 20, 'LineColor', 'none');
hold on;  

% overlay the wall mask as well:

% 1) Define how many vertical “profile” stacks you want across the domain
numProfiles = 30;  % e.g., 20 columns

% 2) Define which columns to use, evenly spaced in the x-direction
colIndices = round(linspace(1, size(X,2), numProfiles));

% 3) (Optional) vertical skip to avoid arrow clutter
verticalSkip = 1;  % if you want every row, use 1; if you want to skip some rows, increase this

% 4) Build list of rows
rowIndices = 1:verticalSkip:size(X,1);

% 5) Quiver arrow scaling: smaller = bigger arrows relative to data
arrowScale = 0.5;

% 6) Plot each “profile” column in black
for c = colIndices
    quiver(...
        X(rowIndices, c)*1000, Y(rowIndices, c)*1000, ...   % positions
        avgU(rowIndices, c), avgV(rowIndices, c), ...  % velocity components
        arrowScale, ...
        'k' ...  % black color for the vectors
    );
end

% % 7) Define a red-to-blue colormap. 
% %    This simple linear interpolation goes from [1, 0, 0] (red) to [0, 0, 1] (blue).
% numColors = 256;
% colors = zeros(numColors, 3);
% % Red channel decreases from 1 to 0
% colors(:,1) = linspace(1, 0, numColors); 
% % Green channel is 0 throughout
% colors(:,2) = 0; 
% % Blue channel increases from 0 to 1
% colors(:,3) = linspace(0, 1, numColors);

% 8) Apply this colormap, set color limits, etc.
% colormap(sky);
colormap(redblue);
caxis([min(avgVel(:)), max(avgVel(:))]);
colorbar;

% 9) Keep aspect ratio square for a proper spatial plot
axis equal;
title('Velocity Profile');
xlabel('X (mm)');
ylabel('Y (mm)');
%
%%

% -----------------------------------------
% Example: Plot u+ vs y+ for boundary-layer
% -----------------------------------------

% Let's assume you have already loaded data (x, y, avgU, etc.)
% from your existing script. If not, do so:
% data = load('yourData.mat');
% X = data.x{1}; Y = data.y{1};  % or however your data is stored
% avgU = ...  (time-averaged velocity component in x-direction)

% USER-PROVIDED / KNOWN PHYSICAL PARAMETERS
% We'll define placeholders that the user must fill in (real numeric values).
rho     = 997.98;        % [kg/m^3]  (Water at 21C)
mu      = 1.0016e-3;      % [Pa*s] dynamic viscosity of fluid (Water at 20C)
% or kinematic viscosity nu = mu / rho (example ~1.48e-5 m^2/s)
nu      = mu / rho;     

u_tau   = 0.5;         % [m/s] friction velocity, user to define (example)
% Typically, u_tau = sqrt(tau_wall / rho), if you know the wall shear stress

% PICK A REFERENCE X LOCATION
% Suppose you want to examine boundary-layer at xIndex.
% For demonstration, pick the center of your domain or near the boundary.
xIndex = 110;  % or any integer in [1, size(X,2)]
% Make sure that you have that many columns in your data.

% Extract the 1D slices at that xIndex:
yVals = Y(:, xIndex);
uVals = avgU(:, xIndex);

% If there's a wall mask, make sure to skip NaN or masked points
validIdx = ~isnan(uVals);
yVals    = yVals(validIdx);
uVals    = uVals(validIdx);

% Optionally, if y=0 is the wall, ensure yVals is measured from the wall.
% If your data's (0,0) is already the wall, you're fine.
% Otherwise, you may need to shift yVals so that the minimum y is 0:
yVals = yVals - min(yVals);

% COMPUTE Y+ AND U+
% We can do this using dynamic viscosity or kinematic viscosity
% For kinematic viscosity approach:  y+ = (u_tau / nu) * y
% For dynamic viscosity approach:    y+ = (rho * u_tau / mu) * y

% Let's pick kinematic viscosity approach
yPlus = (u_tau / nu) .* yVals;

% Now compute u+ 
uPlus = uVals / u_tau;
kappa = 0.41;
B= 5;
% PLOT U+ VS Y+
figure;
% subplot(1,2,1);
plot(yPlus, uPlus, 'ko-','LineWidth',1.5,'MarkerSize',5);
xlabel('y^+');
ylabel('u^+');
title(['Boundary Layer Profile at X index = ', num2str(xIndex)]);
grid on;

% Optionally, use semilog scale on X-axis (common for boundary layer plots)
set(gca,'XScale','log'); 
% set(gca,'YScale','log');  (Typically, we don't log scale the velocity axis.)

% If you want to compare with the log-law region or the linear region of a boundary layer:
% The classical "law of the wall" states: u+ ~ 1/0.41 ln(y+) + 5.0  (roughly),
% so you can overlay a reference line for the log-law:

hold on;
yPlusFit = logspace(0, 3, 100);  % e.g., from y+ = 1 to y+ = 1000
uPlusLogLaw = (1/kappa)*log(yPlusFit) + B; % classic Karman constant=0.41, intercept~5
plot(yPlusFit, uPlusLogLaw, 'r--','LineWidth',1.5, 'DisplayName','Log-law');
xlim([0 1000])
legend({'Data','Log-law'}, 'Location','best');

% (Optional) If you want linear region near y+ < ~5, you might also overlay y+ line:
% A near-wall linear: u+ = y+ (for y+ < ~5)

uPlusLinear = yPlusFit;  % same as y+ if < 5
plot(yPlusFit(yPlusFit<12), uPlusLinear(yPlusFit<12), 'b-.','LineWidth',1.5, 'DisplayName','Linear near-wall');

legend({'Data','Log-law','Linear near-wall'}, 'Location','best');
hold off;

% OPTIONAL: COMPARE MULTIPLE x-LOCATIONS
% If you want to do multiple xIndices in one plot:
% subplot(1,2,2);
figure;
xIndices = [90, 100, 110];  % pick multiple x-locations
markers  = {'o','s','^'}; % different markers
for i = 1:length(xIndices)
    xi = xIndices(i);
    yVals_i = Y(:, xi);
    uVals_i = avgU(:, xi);
    valid   = ~isnan(uVals_i);
    yVals_i = yVals_i(valid);
    uVals_i = uVals_i(valid);

    % Shift so min y is zero if needed
    yVals_i = yVals_i - min(yVals_i);

    yPlus_i = (u_tau / nu) .* yVals_i;
    uPlus_i = uVals_i / u_tau;
    
    % plot each
    semilogx(yPlus_i, uPlus_i, [markers{i}, '-'], ...
        'DisplayName',['xIndex=',num2str(xi)],...
        'LineWidth',1.5, 'MarkerSize',5); 
    hold on;
end
% Overplot the log law again for reference
yPlusFit = logspace(0, 3, 100);
uPlusLogLaw = (1/0.41)*log(yPlusFit) + 5.0;
plot(yPlusFit, uPlusLogLaw, 'k--', 'LineWidth',1.5, 'DisplayName','Log-law');

uPlusLinear = yPlusFit;  % same as y+ if < 5
plot(yPlusFit(yPlusFit<12), uPlusLinear(yPlusFit<12), 'b-.','LineWidth',1.5, 'DisplayName','Linear near-wall');

legend({'Data','Log-law','Linear near-wall'}, 'Location','best');

grid off; axis tight;
box on;
xlabel('y^+');
ylabel('u^+');
title('BL Profiles at Multiple X Indices');
legend('Location','best');
hold off;

%%
% Calculate Turbulence Intensity (TI) Field
% Stack instantaneous u_component frames (assumed to be of the same size)
uStack = cat(3, data.u_component{:});

% Compute the standard deviation of u at each (x,y) position, ignoring NaNs.
stdU = squeeze(nanstd(uStack, 0, 3));

% Calculate turbulence intensity as the ratio of u fluctuations to time-averaged u.
TI_u = stdU ./ abs(avgU);  

% Convert to percentage:
TI_u_percent = TI_u * 100;

% Cap values above 200 to 200
% TI_u_percent(TI_u_percent > 800) = 800;

% Apply wall mask:
TI_u_percent(wall) = NaN;

% Plot Turbulence Intensity
figure('Position', [100 100 800 600]);
contourf(X*1000, Y*1000, TI_u_percent, 25, 'LineColor', 'none');
colormap(jet);
colorbar;
% caxis([0 800]); % Set color axis limits
xlabel('X (mm)');
ylabel('Y (mm)');

% Use LaTeX interpreter for the title
title('Turbulence Intensity %', 'Interpreter', 'latex', 'FontSize', 12);

axis equal;

%%
% Calculate Turbulence Kinetic Energy (TKE)
% Stack instantaneous fields. These steps should match how you computed avgU and avgV.
uStack = cat(3, data.u_component{:});
vStack = cat(3, data.v_component{:});

% Compute the standard deviation (fluctuation level) for each spatial location.
% nanstd is used so that NaNs do not affect the calculations.
stdU = squeeze(nanstd(uStack, 0, 3));  % Standard deviation in u-direction
stdV = squeeze(nanstd(vStack, 0, 3));  % Standard deviation in v-direction

% Calculate TKE for a 2D flow (if you have only two velocity components)
TKE = 0.5 * (stdU.^2 + stdV.^2);

% Optionally, set wall regions to NaN so they don't show up in the plot.
TKE(wall) = NaN;

% Plotting Turbulence Kinetic Energy
figure('Position',[100 100 800 600]);
contourf(X*1000, Y*1000, TKE, 10, 'LineColor', 'none');
colormap(jet);  % You can choose any colormap you prefer
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Turbulence Kinetic Energy, k = 0.5(\sigma_u^2 + \sigma_v^2)', 'Interpreter', 'tex');
axis equal;
%
%---- Mean Velocity Profiles vs. y ----%
% Pick a set of x-indices (or positions) across your domain.
xIndices = round(linspace(1, size(X,2), 3));  % now 7 profiles
figure;
hold on;
markers = {'o','s','^','d','v'};  % Only 5 markers defined
nMarkers = length(markers);
for i = 1:length(xIndices)
    xi = xIndices(i);
    % Extract the y-location and corresponding u-velocity profile.
    yProfile = Y(:, xi);
    uProfile = avgU(:, xi);
    
    % Remove NaN values (e.g., if wall regions or masked zones exist)
    valid = ~isnan(uProfile);
    yProfile = yProfile(valid);
    uProfile = uProfile(valid);
    
    % Shift y so that the wall is at y = 0 (if necessary)
    yProfile = yProfile - min(yProfile);
    
    % Wrap around marker index using modulo operator.
    markerInd = mod(i-1, nMarkers) + 1;
    
    plot(uProfile, yProfile*1000, markers{markerInd}, 'LineWidth', 1.5, ...
         'DisplayName', ['x-index = ' num2str(xi)]);
end
xlabel('Time-averaged u (m/s)');
ylabel('y (mm)');
title('Mean Velocity Profiles at Selected x-locations');
legend('show','Location','best');
grid off;
box on;
hold off;


% Reynolds Stress and Normal Stress Components Computation

% Stack instantaneous u- and v-component frames.
uStack = cat(3, data.u_component{:});
vStack = cat(3, data.v_component{:});
numFrames = size(uStack, 3);

% Preallocate arrays for the Reynolds stresses
% It is more efficient to compute these fields by subtracting the time-average 
% from the instantaneous fields and then averaging the products over time.
% We can use nanmean for this purpose.
uPrime  = bsxfun(@minus, uStack, avgU);
vPrime  = bsxfun(@minus, vStack, avgV);

% Compute the statistics at each (x,y) location ignoring NaNs:
uuStress = squeeze(nanmean(uPrime.^2, 3));  % <u'u'>
vvStress = squeeze(nanmean(vPrime.^2, 3));  % <v'v'>
uvStress = squeeze(nanmean(uPrime.*vPrime, 3));  % <u'v'>

% Apply the wall mask so that masked regions are not plotted.
uuStress(wall) = NaN;
vvStress(wall) = NaN;
uvStress(wall) = NaN;

% Plotting Reynolds Stress Fields

% figure('Position',[150 150 1200 400]);

%--- Plot <u'u'> ---
figure('Position',[150 150 1200 400]);
% subplot(1,3,1);
contourf(X*1000, Y*1000, uuStress, 20, 'LineColor', 'none');
colormap(gca, parula);   % Use a preferred colormap, here parula is chosen
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Reynolds Normal Stress: <u''u''>'); 
axis equal;

%--- Plot <v'v'> ---
figure('Position',[150 150 1200 400]);
% subplot(1,3,2);
contourf(X*1000, Y*1000, vvStress, 20, 'LineColor', 'none');
colormap(gca, parula);
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Reynolds Normal Stress: <v''v''>'); 
axis equal;

%--- Plot <u'v'> ---
figure('Position',[150 150 1200 400]);
% subplot(1,3,3);
contourf(X*1000, Y*1000, uvStress, 20, 'LineColor', 'none');
colormap(gca, parula);
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Reynolds Shear Stress: <u''v''>'); 
axis equal;

% sgtitle('Reynolds Stress Components');


%
%---- Boundary-layer integral parameters at a given x-location ----%
xIndex = round(size(X,2)/2); % choose an x-location (e.g., mid-domain)
yProfile = Y(:, xIndex);
uProfile = avgU(:, xIndex);

% Remove NaN and shift y if necessary
valid = ~isnan(uProfile);
yProfile = yProfile(valid);
uProfile = uProfile(valid);
yProfile = yProfile - min(yProfile);

% Define the free-stream (edge) velocity, U_infty. One option is the maximum value.
U_inf = max(uProfile);

% Compute displacement thickness, delta*:
%   \delta^* = \int_0^\infty \left[1 - \frac{u(y)}{U_\infty}\right] dy
delta_star = trapz(yProfile, (1 - uProfile./U_inf));

% Compute momentum thickness, theta:
%   \theta = \int_0^\infty \frac{u(y)}{U_\infty} \left[1 - \frac{u(y)}{U_\infty}\right] dy
theta = trapz(yProfile, (uProfile./U_inf).*(1 - uProfile./U_inf));

% Shape factor, H:
H = delta_star/theta;

% Display the values in the command window:
fprintf('At x-index %d:\n', xIndex);
fprintf('  Displacement thickness, delta* = %.4f m\n', delta_star);
fprintf('  Momentum thickness, theta = %.4f m\n', theta);
fprintf('  Shape factor, H = %.4f\n', H);

%
%---- Compute Vorticity Field ----%
% Compute scalar grid spacings assuming uniform spacing:
dx = mean(diff(X(1,:)));  % average spacing in the x-direction
dy = mean(diff(Y(:,1)));  % average spacing in the y-direction

% Use MATLAB's gradient function on avgU and avgV:
% Here, the second argument gives the x-spacing, and the third gives the y-spacing.
[dv_dx, ~] = gradient(avgV, dx, dy);
[~, du_dy] = gradient(avgU, dx, dy);

% Compute vorticity (omega_z)
vorticity = dv_dx - du_dy;

% Set wall masked regions to NaN so they don't appear in the plot
vorticity(wall) = NaN;

% Plot the vorticity field
figure('Position', [200 200 800 600]);
contourf(X*1000, Y*1000, vorticity, 30, 'LineColor', 'none');
% Use a colormap; if you don't have 'cool', you can try 'parula', 'jet', etc.
colormap('hot');
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Vorticity, \omega_z = \partial v/\partial x - \partial u/\partial y');
axis equal;

%
% Interactive PSD and Energy Spectrum Computation for Velocity Fluctuations
% (Assumes you have already loaded data, computed avgU, avgVel, and defined X & Y.)

% -------------------------------
% STEP 1: Let the user select locations
% -------------------------------
figure('Name','Select Locations for PSD','Position',[100 100 800 600]);
contourf(X, Y, avgVel, 20, 'LineColor','none');
colormap(parula);
axis equal;
colorbar;
xlabel('X (mm)');
ylabel('Y (mm)');
title('Click on locations for PSD. Press Enter when finished.');
hold on;
[clickX, clickY] = ginput;  % User clicks multiple times; press Enter when done.
plot(clickX, clickY, 'kx','MarkerSize',10,'LineWidth',2);
hold off;

if isempty(clickX)
    disp('No points were selected.');
    return;
end
%
% -------------------------------
% STEP 2: Initialize Figures for PSD
% -------------------------------

% Initialize the loglog figure once
hLoglog = figure('Name','Energy Spectrum (loglog)','Position',[250 250 800 600]);
hold on;
title('Power Spectral density of u');
xlabel('Frequency (Hz)');
ylabel('PSD ((m/s)^2/Hz)');
grid on;

% -------------------------------
% STEP 3: Parameters for PSD Calculation
% -------------------------------
fs = 267647;               % Sampling frequency (Hz); replace with your actual value.
N_length = length(data.u_component);
windowLength = 16;    % Window length for pwelch
window = hanning(N_length/windowLength)';
noverlap = round(0.5 * windowLength);
nfft = [];            % Let pwelch choose or specify a value.
numPoints = length(clickX);
colors = lines(numPoints);  % To assign different colors per curve

% -------------------------------
% STEP 4: Loop Over Each Selected Point
% -------------------------------

for i = 1:numPoints
        % Find the nearest indices from the click locations
    [~, xIndex] = min(abs(X(1,:) - clickX(i)));
    [~, yIndex] = min(abs(Y(:,1) - clickY(i)));
    
    fprintf('Selected point %d at (xIndex, yIndex) = (%d, %d)\n', i, xIndex, yIndex);
    
    % Extract time series for the u-component at that point.
    numFrames = numel(data.u_component);
    uTimeSeries = nan(numFrames,1);
    for k = 1:numFrames
        frame = data.u_component{k};
        uTimeSeries(k) = frame(yIndex, xIndex);
    end
    
    % Remove NaNs if present.
    valid = ~isnan(uTimeSeries);
    if ~any(valid)
        warning('No valid data at the selected point %d. Skipping.', i);
        continue;
    end
    uTimeSeries = uTimeSeries(valid);
    
    % Subtract the local mean (from avgU) to obtain fluctuations
    localMean = avgU(yIndex, xIndex);
    uFluct = uTimeSeries - localMean;
    
    % Compute fluctuations and PSD using pwelch
    [Pxx, f] = pwelch(uFluct, window, noverlap, nfft, fs);
    
    % Make sure f and Pxx contain positive values for log scale
    validIdx = f > 0 & Pxx > 0;
    if any(validIdx)
        figure(hLoglog);  % Ensure we use the same loglog figure
        loglog(f(validIdx), Pxx(validIdx)', 'Color', colors(i,:), 'LineWidth', 1, ...
               'DisplayName', sprintf('(%d, %d)', yIndex, xIndex));
    else
        warning('Point %d produced non-positive values for f or Pxx. Skipping loglog plot.', i);
    end
end

% Force the axes to remain loglog, in case of any interference
figure(hLoglog);
set(gca, 'XScale', 'log', 'YScale', 'log');
box on;
legend('Location','best');


%% === PART A: Extract and Save Velocity Profile Lines (Throat and Right Only) ===
% (Assumes you have computed avgVel, and defined grids X and Y.)
%
% This section:
% 1. Displays the avgVel contour plot and lets the user select the throat location.
% 2. Asks for horizontal spacing (in pixels) and number of extraction lines.
% 3. Extracts vertical profiles (columns) of avgVel from the throat location to the right.
% 4. Shows the extraction lines in red on a separate figure.
% 5. Saves a MAT file containing the extraction indices, profiles, throat location, and grid.

% --- Step 1: Display the contour and select throat location ---
figure('Name','Select Throat Location','Position',[100 100 800 600]);
contourf(X*1000, Y*1000, avgVel, 20, 'LineColor','none');
colormap(parula);
axis equal; colorbar;
xlabel('X (mm)'); 
ylabel('Y (mm)');
title('Click on the throat location for extraction. Press Enter when done.');
hold on;
[throatX, throatY] = ginput(1);
plot(throatX, throatY, 'kx','MarkerSize',12,'LineWidth',2);
hold off;
pause(0.5);

% --- Step 2: Ask for extraction parameters ---
prompt   = {'Enter horizontal spacing (in pixels) between profiles:', ...
            'Enter number of extraction lines (to extract from the throat to the right):'};
dlgtitle = 'Extraction Parameters';
dims     = [1 50];
definput = {'10','5'};
answer   = inputdlg(prompt, dlgtitle, dims, definput);
hSpacing = str2double(answer{1});
numLines = str2double(answer{2});

% --- Step 3: Determine extraction indices (throat and to the right only) ---
[~, throatIndex] = min(abs(X(1,:)*1000 - throatX));
extractionIndices = throatIndex + (0:numLines-1)*hSpacing;
% Ensure indices are within the valid range.
extractionIndices = extractionIndices(extractionIndices<=size(X,2));

% --- Step 4: Extract vertical profiles at those indices ---
numExtract = length(extractionIndices);
profiles = cell(numExtract,1);
profiles_u = cell(numExtract,1);
profiles_v = cell(numExtract,1);
for i = 1:numExtract
    profiles{i} = avgVel(:, extractionIndices(i));  % one column from avgVel
    profiles_u{i} = avgU(:, extractionIndices(i));
    profiles_v{i} = avgV(:, extractionIndices(i));

end

% --- Step 5: Show extraction lines on a separate figure ---
figure('Name','Extraction Lines on avgVel','Position',[100 100 900 600]);
contourf(X*1000, Y*1000, avgVel, 20, 'LineColor','none');
colormap(parula);
axis equal; colorbar;
xlabel('X (mm)'); 
ylabel('Y (mm)');
title('Average Velocity with Extraction Lines (Red)');
hold on;
for i = 1:numExtract
    % Draw a red vertical line at the given x-index:
    xVal = X(1, extractionIndices(i))*1000;
    % Plot a line spanning the full y-range of Y.
    plot([xVal xVal], [min(Y(:))*1000 max(Y(:))*1000], 'r-', 'LineWidth', 2);
end
hold off;

% --- Step 6: Save extracted profiles ---
dataToSave = struct;
dataToSave.extractionIndices = extractionIndices;  % the x-indices (first saved field)
dataToSave.profiles = profiles;
dataToSave.profiles_u= profiles_u;
dataToSave.profiles_v= profiles_v;
dataToSave.throat = [throatX, throatY];
dataToSave.X = X;
dataToSave.Y = Y;

savePrompt = {'Enter file name for saving the extracted profiles (without extension):'};
saveDlgTitle = 'Save Profiles';
saveDims = [1 50];
saveDefInput = {'velocityProfiles'};
saveFileName = inputdlg(savePrompt, saveDlgTitle, saveDims, saveDefInput);
if isempty(saveFileName)
    error('No file name provided. Aborting.');
end
matFileName = [saveFileName{1} '.mat'];
save(matFileName, 'dataToSave');
fprintf('Saved %d profile lines at indices %s to %s\n', numExtract, mat2str(extractionIndices), matFileName);

%% === PART B: Load and Compare Extracted Velocity Profiles from MAT Files ===
% This script lets you select one or more MAT files (saved from Part A)
% and then plots the extracted velocity profiles. Each extraction line is
% displayed in its own panel. The first panel is labeled "Throat" and the
% other panels show the distance (in mm) from the throat (formatted with one decimal place).
% The x-axis tick labels are formatted to show at most one decimal place.
% A common legend (using file names without the ".mat" extension) appears on the east side of the figure.

% --- Step 1: Select MAT files ---
[fileNames, filePath] = uigetfile('*.mat', 'Select Velocity Profile Files', 'MultiSelect', 'on');
if isnumeric(fileNames)
    disp('No files selected. Exiting.');
    return;
end
if ~iscell(fileNames)
    fileNames = {fileNames};
end

% --- Step 2: Read the first file to determine the number of extraction lines ---
firstFile = fullfile(filePath, fileNames{1});
loadedData = load(firstFile);
dataIn = loadedData.dataToSave;
numPanels = length(dataIn.extractionIndices);
% Retrieve the throat x coordinate (in mm) from the data.
throatX = dataIn.throat(1);
% Preallocate cell arrays to hold profile data from each file for each extraction line.
% profilesCollection{j, i} is the j-th extracted profile from file i.
profilesCollection = cell(numPanels, length(fileNames));
legendEntries = cell(1, length(fileNames));

% --- Step 3: Loop over each file to extract the profiles ---
for i = 1:length(fileNames)
    fullFileName = fullfile(filePath, fileNames{i});
    loadedData = load(fullFileName);
    dataIn = loadedData.dataToSave;
    % Remove the ".mat" extension using fileparts.
    [~, name, ~] = fileparts(fileNames{i});
    legendEntries{i} = name;  % Use file name without ".mat" as the legend entry.
    
    % Check if the number of extraction lines matches:
    if length(dataIn.extractionIndices) ~= numPanels
        warning('File %s has a different number of extraction lines. Skipping it.', fileNames{i});
        continue;
    end
    
    for j = 1:numPanels
        profilesCollection{j, i} = dataIn.profiles{j};  % each profile is a vector.
    end
    
    % Store a common y-grid (assumed to be a column vector in meters).
    commonY = dataIn.Y(:,1);
end

% --- Step 4: Create a high-quality landscape figure with tiled panels ---
figure('Name','Comparison of Velocity Profiles','Position',[100 100 1600 600]);
% Use 'loose' padding to allow space for axis titles.
tl = tiledlayout(1, numPanels, 'TileSpacing','compact', 'Padding','loose');
tl.TileIndexing = 'rowmajor';

% Define high-quality fonts and line widths.
titleFontSize = 14;
labelFontSize = 12;
tickFontSize  = 10;
lineWidthVal  = 2;

% Choose a set of bright colors for the different files.
colors = lines(length(fileNames));
% Define a set of markers. If more files than markers, the markers will repeat.
markers = {'o', 's', 'd', '^', 'v', '>', '<', 'p', 'h'};

% Preallocate a vector for storing plot handles from the first tile (for legend).
fileHandles = gobjects(1, length(fileNames));

% --- Step 5: Plot each extraction line in its own panel ---
for j = 1:numPanels
    ax = nexttile;
    hold on;
    
    % Compute the distance (in mm) from the throat.
    % xVal: x-coordinate (in mm) corresponding to the j-th extraction index.
    xVal = dataIn.X(1, dataIn.extractionIndices(j));
    distanceFromThroat = xVal - throatX/1000;  % in mm
    
    % For each file, plot the corresponding profile.
    % Note: Convert commonY from meters to mm by multiplying by 1000.
    for i = 1:length(fileNames)
        currentProfile = profilesCollection{j, i};
        if isempty(currentProfile)
            continue;
        end
        
        % Determine marker (cycle through markers if there are more files than markers)
        markerIdx = mod(i-1, length(markers)) + 1;
        % Plot the profile vs. commonY (converted to mm)
        hPlot = plot(currentProfile, commonY*1000, 'LineWidth', lineWidthVal, ...
            'Color', colors(i,:), 'Marker', markers{markerIdx}, 'MarkerSize', 2.5, ...
            'DisplayName', legendEntries{i});
        % For the first panel, store each file's plot handle.
        if j == 1 && isempty(fileHandles(i))
            fileHandles(i) = hPlot;
        end
    end
    
    % Set the title: for panel 1, display "Throat", otherwise show the distance.
    if j == 1
        titleStr = 'Throat';
    else
        titleStr = sprintf('%.1f mm', distanceFromThroat*1000);
    end
    title(titleStr, 'FontSize', titleFontSize);
    xlabel('Velocity (m/s)', 'FontSize', labelFontSize);
    ylabel('Y (mm)', 'FontSize', labelFontSize);  % Updated label for mm
    
    % Adjust x-axis limits to fit data snugly.
    ax.FontSize = tickFontSize;
    % Let MATLAB choose x-limits based on the data:
    xLimits = ax.XLim;
    % Force 4 ticks on x-axis with labels formatted to one decimal place.
    xticks = linspace(xLimits(1), xLimits(2), 6);
    ax.XTick = xticks;
    ax.XTickLabel = arrayfun(@(v) sprintf('%.1f', v), xticks, 'UniformOutput', false);
    
    % Set y-axis limits for all panels (converted to mm).
    ylim([0.15, 0.75]);
    
    set(gca, 'Box', 'on', 'XGrid', 'off', 'YGrid', 'off');
    hold off;
end

% --- Step 6: Add a common legend outside in a column format ---
lgd = legend('show', 'Location', 'northeastoutside');
lgd.FontSize = 8;
lgd.NumColumns = 1;  % Display legend entries in a single column

% % Optionally, add an overall title to the entire tiled layout.
% tl.Title.String = 'Comparison of Extracted Velocity Profiles';
% tl.Title.FontSize = titleFontSize + 2;
%%
%% === PART B: Load and Compare Extracted Velocity Profiles from MAT Files ===
% This script lets you select one or more MAT files (saved from Part A)
% and then plots velocity profiles from each file. 
% Key features:
%   1) Each file's data is referenced to its own throat location.
%   2) A single legend on the far-right edge of the tiled plot.
%   3) The user can specify a range/list of extraction lines to plot 
%      (e.g., "1 3 5" or "1:2:9").
%   4) Optional different line styles for each file in addition to 
%      distinct colors and markers.

clear; clc; close all;

% -------------------------------
% Step 1: Select MAT files
% -------------------------------
[fileNames, filePath] = uigetfile('*.mat', ...
    'Select Velocity Profile Files', 'MultiSelect', 'on');
if isnumeric(fileNames)
    disp('No files selected. Exiting.');
    return;
end
if ~iscell(fileNames)
    fileNames = {fileNames};
end

% -------------------------------
% Step 2: Read the first file to check how many total lines exist
% -------------------------------
firstFile = fullfile(filePath, fileNames{1});
loadedData = load(firstFile);
dataIn     = loadedData.dataToSave;

% We assume all files have the same number of extraction lines and the same Y grid.
numTotalLines = length(dataIn.extractionIndices);

% For safety, verify that all files do indeed share the same Y dimension:
commonY_first = dataIn.Y(:,1);
for i = 2:length(fileNames)
    checkData = load(fullfile(filePath, fileNames{i}));
    thisY     = checkData.dataToSave.Y(:,1);
    if ~isequal(thisY, commonY_first)
        warning('File "%s" has a different Y grid. Inconsistent data!', fileNames{i});
        % You can decide whether to continue or return here.
    end
end
commonY = commonY_first; % (meters)

% -------------------------------
% Step 3: Ask user for which lines to plot
% -------------------------------
fprintf('There are %d total extraction lines.\n', numTotalLines);
fprintf('Enter the line indices you want to display.\n');
fprintf('Examples: "1 4 5 9" or "1:2:9" or just "1" etc.\n');

% Prompt the user for input
userInput = input('Enter extraction line indices: ', 's');
if isempty(userInput)
    disp('No input provided. Exiting.');
    return;
end

% Convert string to numeric array (MATLAB can handle "4:2:12" etc.)
selectedLineIdx = str2num(userInput); %#ok<ST2NM> 
% Ensure we only take valid indices in [1, numTotalLines].
selectedLineIdx = selectedLineIdx(selectedLineIdx >= 1 & selectedLineIdx <= numTotalLines);
selectedLineIdx = unique(selectedLineIdx);  % remove duplicates, sort ascending

if isempty(selectedLineIdx)
    disp('No valid extraction line indices given. Exiting.');
    return;
end

numPlotLines = length(selectedLineIdx);
disp(['Selected line indices: ', num2str(selectedLineIdx)]);

% -------------------------------
% Step 4: Preallocate storage
% -------------------------------
% We'll store the main, profiles_u, and profiles_v from each file.
% profiles_main{j, i} = dataIn.profiles{j} from file i
profiles_main = cell(numTotalLines, length(fileNames));
profiles_u    = cell(numTotalLines, length(fileNames));
profiles_v    = cell(numTotalLines, length(fileNames));

% Also store the X-coordinates for each line from each file
xCoord_mm = zeros(numTotalLines, length(fileNames)); 
% We'll store the throat for each file as well
throatX_mm = zeros(1, length(fileNames));
throatY_m  = zeros(1, length(fileNames));

% We'll build a legend using the file names
legendEntries = cell(1, length(fileNames));

% -------------------------------
% Step 5: Load each file, store data
% -------------------------------
for i = 1:length(fileNames)
    fullFileName = fullfile(filePath, fileNames{i});
    loadedData   = load(fullFileName);
    dataIn       = loadedData.dataToSave;
    
    [~, name, ~]  = fileparts(fileNames{i});
    legendEntries{i} = name;  % File name without .mat
    
    % Check # extraction lines
    if length(dataIn.extractionIndices) ~= numTotalLines
        warning('File %s has a different # of extraction lines. Skipping.', fileNames{i});
        continue;
    end
    
    % Store the throat location (in mm for X, in m for Y)
    throatX_mm(i) = dataIn.throat(1)*1000; 
    throatY_m(i)  = dataIn.throat(2);
    
    % Store main velocity profiles
    for j = 1:numTotalLines
        profiles_main{j,i} = dataIn.profiles{j};
    end
    
    % If profiles_u exist
    if isfield(dataIn, 'profiles_u')
        for j = 1:numTotalLines
            profiles_u{j,i} = dataIn.profiles_u{j};
        end
    end
    
    % If profiles_v exist
    if isfield(dataIn, 'profiles_v')
        for j = 1:numTotalLines
            profiles_v{j,i} = dataIn.profiles_v{j};
        end
    end
    
    % Store the X-coordinates (in mm) for each extraction line
    for j = 1:numTotalLines
        x_mm = dataIn.X(1, dataIn.extractionIndices(j)) * 1000; 
        xCoord_mm(j,i) = x_mm;
    end
end

% -------------------------------
% Helper function to build a figure & tiled layout
% -------------------------------
makeProfileFigure = @(figName, nTiles) figure('Name', figName, ...
    'Position',[100 100 1600 600], 'Color','w');

makeTileLayout = @(nTiles) tiledlayout(1, nTiles, 'TileSpacing','compact', 'Padding','compact');

% -------------------------------
% Define style constants
% -------------------------------
titleFontSize = 14;
labelFontSize = 12;
tickFontSize  = 10;
lineWidthVal  = 2;

% Colors, markers, and line styles (user can comment out line styles if desired)
colors     = lines(length(fileNames));
markers    = {'o','s','d','^','v','>','<','p','h'};
lineStyles = {'-','--',':','-.'};  % comment out if not desired

% -------------------------------------------------------------------------
% Function to plot a set of profiles (like main, U, or V) in a new figure
% -------------------------------------------------------------------------


% ------------------------------------------------------
% PART A: PLOT THE MAIN VELOCITY PROFILES
% ------------------------------------------------------
plotProfilesInFigure('Comparison of Main Velocity Profiles', ...
    profiles_main, ...
    fileNames, legendEntries, ...
    selectedLineIdx, ...
    xCoord_mm, throatX_mm, throatY_m, ...
    commonY, ...
    colors, markers, lineStyles, ...
    titleFontSize, labelFontSize, tickFontSize, lineWidthVal);

% ------------------------------------------------------
% PART B: PLOT THE AVERAGE U PROFILES (profiles_u)
% ------------------------------------------------------
hasProfilesU = any(cellfun(@(x) ~isempty(x), profiles_u(:)));
if hasProfilesU
    plotProfilesInFigure('Comparison of Average U Velocity Profiles', ...
        profiles_u, ...
        fileNames, legendEntries, ...
        selectedLineIdx, ...
        xCoord_mm, throatX_mm, throatY_m, ...
        commonY, ...
        colors, markers, lineStyles, ...
        titleFontSize, labelFontSize, tickFontSize, lineWidthVal);
end

% ------------------------------------------------------
% PART C: PLOT THE AVERAGE V PROFILES (profiles_v)
% ------------------------------------------------------
hasProfilesV = any(cellfun(@(x) ~isempty(x), profiles_v(:)));
if hasProfilesV
    plotProfilesInFigure('Comparison of Average V Velocity Profiles', ...
        profiles_v, ...
        fileNames, legendEntries, ...
        selectedLineIdx, ...
        xCoord_mm, throatX_mm, throatY_m, ...
        commonY, ...
        colors, markers, lineStyles, ...
        titleFontSize, labelFontSize, tickFontSize, lineWidthVal);
end

disp('Done!');


%% Save All Open Figures as PNG Files (300dpi, minimal white space)

% Prompt the user to select a folder to save the images.
folderName = uigetdir(pwd, 'Select a folder to save all open figures');
if folderName == 0
    disp('User cancelled folder selection. Exiting.');
    return;
end

% Get the screen size for maximizing figures
screenSize = get(0, 'ScreenSize'); % [left, bottom, width, height]

% Find all open figure handles.
figHandles = findall(0, 'Type', 'figure');

% Loop over each figure and save it.
for k = 1:length(figHandles)
    fig = figHandles(k);

    % Save the original position of the figure
    originalPosition = fig.Position;

    % Maximize the figure to fill the screen
    fig.Position = screenSize;
    drawnow; % Ensure the figure updates to the maximized size

    % Retrieve the axes handle for exporting
    ax = findall(fig, 'Type', 'axes');
    if ~isempty(ax)
        % Export the axes contents (removes extra white space)
        fileName = fullfile(folderName, sprintf('Figure_%d.png', fig.Number));
        exportgraphics(ax(1), fileName, 'Resolution', 300, 'BackgroundColor', 'none');
    else
        % If no axes found, save the entire figure
        fileName = fullfile(folderName, sprintf('Figure_%d.png', fig.Number));
        exportgraphics(fig, fileName, 'Resolution', 300, 'BackgroundColor', 'none');
    end

    % Restore the original figure size
    fig.Position = originalPosition;

    fprintf('Saved figure %d as:\n%s\n', fig.Number, fileName);
end

disp('All open figures have been saved.');




%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                            %Ignore%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Save All Open Figures as PNG Files (300dpi, minimal white space)

% Prompt the user to select a folder to save the images.
folderName = uigetdir(pwd, 'Select a folder to save all open figures');
if folderName == 0
    disp('User cancelled folder selection. Exiting.');
    return;
end

% Find all open figure handles.
figHandles = findall(0, 'Type', 'figure');

% Loop over each figure and save it.
for k = 1:length(figHandles)
    fig = figHandles(k);

    % Bring the figure to the front (optional).
    figure(fig);

    % Retrieve the axes handle for exporting; if multiple axes exist,
    % you might want to export the entire figure.
    % Here we assume one axes per figure.
    ax = findall(fig, 'Type', 'axes');
    if ~isempty(ax)
        % Export the axes contents (this removes extra white space around the axes)
        % Set the background to 'none' if you want a transparent background.
        fileName = fullfile(folderName, sprintf('Figure_%d.png', fig.Number));
        exportgraphics(ax(1), fileName, 'Resolution',300, 'BackgroundColor','none');
    else
        % If no axes found, save the entire figure
        fileName = fullfile(folderName, sprintf('Figure_%d.png', fig.Number));
        exportgraphics(fig, fileName, 'Resolution',300, 'BackgroundColor','none');
    end

    fprintf('Saved figure %d as:\n%s\n', fig.Number, fileName);
end

disp('All open figures have been saved.');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                            %functions%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotProfilesInFigure(figTitle, allProfiles, ...
    fileNamesLocal, legendEntriesLocal, ...
    selectedLineIdxLocal, ...
    xCoordAll_mm, throatXAll_mm, throatYAll_m, ...
    commonY_m, ...
    colorsLocal, markersLocal, lineStylesLocal, ...
    titleFontSizeLocal, labelFontSizeLocal, tickFontSizeLocal, lineWidthValLocal)

    nFiles    = length(fileNamesLocal);
    nPlotLines = length(selectedLineIdxLocal);
    
    % Create figure and tile layout
    figH = makeProfileFigure(figTitle, nPlotLines);
    tl   = makeTileLayout(nPlotLines);
    tl.TileIndexing = 'rowmajor';

    % We'll store one handle per file (for the legend)
    fileHandles = gobjects(1, nFiles);

    for idxTile = 1:nPlotLines
        j = selectedLineIdxLocal(idxTile);  % actual extraction line index
        ax = nexttile;
        hold(ax, 'on');

        % We don't have a single "distance from throat" for all files, 
        % because each file has its own throat. We'll decide how to label 
        % each tile. For consistency, let's label the first tile "Throat", 
        % and others as "Line j" or we can label nothing. 
        % Alternatively, we can label with the distance from the *first file* 
        % but that might be misleading. We'll just call them "Line j" here 
        % (except if j == 1, we call it "Throat"). 
        if j == 1
            titleStr = 'Throat';
        else
            titleStr = sprintf('Line %d', j);
        end
        title(ax, titleStr, 'FontSize', titleFontSizeLocal);

        xlabel(ax, 'Velocity (m/s)', 'FontSize', labelFontSizeLocal);
        ylabel(ax, 'Y from throat (mm)', 'FontSize', labelFontSizeLocal);
        
        set(ax, 'FontSize', tickFontSizeLocal, 'Box','on');

        % For each file, plot the j-th profile
        for iFile = 1:nFiles
            prof = allProfiles{j, iFile};
            if isempty(prof), continue; end

            % SHIFT the Y data by each file's throat. 
            % If commonY_m is in meters, then 
            %   yRelative_mm = (commonY_m - throatYAll_m(iFile)) * 1000.
            yRelative_mm = (commonY_m - throatYAll_m(iFile)) * 1000;

            % We'll keep X as velocity on the X-axis. 
            % "prof" should be the velocity values. 
            % So the user sees velocity vs. yRelative. 

            % Let's pick color, marker, line style, all based on iFile
            colorIdx    = iFile; 
            markerIdx   = mod(iFile-1, length(markersLocal)) + 1;
            lineStyleIdx= mod(iFile-1, length(lineStylesLocal)) + 1;

            hPlot = plot(ax, prof, yRelative_mm, ...
                'Color',    colorsLocal(colorIdx,:), ...
                'Marker',   markersLocal{markerIdx}, ...
                'LineStyle',lineStylesLocal{lineStyleIdx}, ...  % comment out if desired
                'LineWidth', lineWidthValLocal, ...
                'MarkerSize', 3, ...
                'DisplayName', legendEntriesLocal{iFile});

            % Store handle for legend from the first tile only 
            % (that ensures 1 handle per file)
            if idxTile == 1 && isgraphics(hPlot)
                fileHandles(iFile) = hPlot;
            end
        end

        hold(ax, 'off');
    end

    % Attach a single legend to the entire tile layout, on the far right
    % (northeastoutside). In modern MATLAB, we can do:
    lgd = legend(fileHandles, legendEntriesLocal, 'Location','northeastoutside');
    % Alternatively:  legend(tl, fileHandles, legendEntriesLocal, 'Location','northeastoutside');
    lgd.FontSize  = 8;
    lgd.NumColumns = 1;
end
