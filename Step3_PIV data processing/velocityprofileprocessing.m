%% PIV_processing_code.m
% =========================================================================
% This script:
%  1) Lets you select one or more MAT files containing velocity profile data.
%  2) Asks you which extraction lines to plot (e.g., "1 4 6 8" or "1:2:12").
%  3) For each extraction line, computes the "distance from throat" by
%     taking the difference: (throatX_mm(iFile) - xCoord_mm(jLine, iFile)) 
%     for each file iFile, then averaging over valid files.
%  4) The tile title displays the MEAN distance from throat (in mm).
%  5) The command prompt prints:
%       distance (mm) ± standard deviation (mm)
%     for each line (across files).
%  6) Plots three sets of profiles (if they exist):
%       (a) Main velocity profiles    (dataIn.profiles{j})
%       (b) Average U-velocity        (dataIn.profiles_u{j})
%       (c) Average V-velocity        (dataIn.profiles_v{j})
%  7) A single legend is placed on the far-right side ('northeastoutside').
%  8) Different colors, markers, and optional line styles distinguish files.
%
% REQUIREMENTS:
%  - MATLAB R2016b or newer (local functions in scripts).
%  - The loaded MAT files must each have a structure 'dataToSave' with:
%       dataToSave.extractionIndices
%       dataToSave.profiles{j}
%       dataToSave.X
%       dataToSave.Y
%       dataToSave.throat = [throatX, throatY]
%    Optionally: dataToSave.profiles_u{j}, dataToSave.profiles_v{j}
%
% HOW TO RUN:
%  1) Save this entire code as "PIV_processing_code.m".
%  2) Run it in the same directory as your .mat files.
%  3) Follow on-screen prompts.
% =========================================================================

clear; clc; close all;

%% -------------------------------
%  STEP 1: Select MAT files
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

%% -------------------------------
%  STEP 2: Check total # lines and common Y
% -------------------------------
firstFile   = fullfile(filePath, fileNames{1});
loadedData  = load(firstFile);
dataInFirst = loadedData.dataToSave;

numTotalLines = length(dataInFirst.extractionIndices);
commonY_first = dataInFirst.Y(:,1);  % (meters)

% Make sure all files share the same dimension for Y
for iFile = 2:length(fileNames)
    checkData = load(fullfile(filePath, fileNames{iFile}));
    Ytemp     = checkData.dataToSave.Y(:,1);
    if ~isequal(Ytemp, commonY_first)
        warning('File "%s" has a different Y grid. Data might be inconsistent!', fileNames{iFile});
        % You can decide to break or skip, but we'll just continue here.
    end
end

commonY = commonY_first;  % treat this as the 'base' Y

%% -------------------------------
%  STEP 3: Ask which lines to plot
% -------------------------------
fprintf('There are %d total extraction lines.\n', numTotalLines);
fprintf('Enter the line indices you want to display.\n');
fprintf('Examples:\n');
fprintf('  "1 4 5 9"    (explicit list)\n');
fprintf('  "1:2:12"     (MATLAB range notation)\n');
fprintf('  "1"          (just line #1)\n\n');

userInput = input('Enter extraction line indices: ','s');
if isempty(userInput)
    disp('No input. Exiting.');
    return;
end

selectedLineIdx = str2num(userInput); %#ok<ST2NM>
% Keep only valid indices
selectedLineIdx = selectedLineIdx(selectedLineIdx >= 1 & selectedLineIdx <= numTotalLines);
selectedLineIdx = unique(selectedLineIdx);  % remove duplicates, sort ascending
if isempty(selectedLineIdx)
    disp('No valid indices. Exiting.');
    return;
end

numPlotLines = length(selectedLineIdx);
disp(['Selected line indices: ', num2str(selectedLineIdx)]);

%% -------------------------------
%  STEP 4: Preallocate data arrays
% -------------------------------
profiles_main = cell(numTotalLines, length(fileNames));
profiles_u    = cell(numTotalLines, length(fileNames));
profiles_v    = cell(numTotalLines, length(fileNames));

throatX_mm = zeros(1, length(fileNames));
throatY_m  = zeros(1, length(fileNames));

% We'll store X-coord in mm for each line from each file
xCoord_mm  = zeros(numTotalLines, length(fileNames));

legendEntries = cell(1, length(fileNames));

%% -------------------------------
%  STEP 5: Load each file and store relevant data
% -------------------------------
for i = 1:length(fileNames)
    fullName = fullfile(filePath, fileNames{i});
    loaded   = load(fullName);
    dataIn   = loaded.dataToSave;
    
    [~, shortName, ~] = fileparts(fileNames{i});
    legendEntries{i}   = shortName;
    
    if length(dataIn.extractionIndices) ~= numTotalLines
        warning('File "%s" has different # extraction lines; skipping.', fileNames{i});
        continue;
    end
    
    % Store throat location
    throatX_mm(i) = dataIn.throat(1)*1000;  % X in mm
    throatY_m(i)  = dataIn.throat(2);      % Y in m
    
    % Store main profiles
    for j = 1:numTotalLines
        profiles_main{j,i} = dataIn.profiles{j};
    end
    
    % Store profiles_u if available
    if isfield(dataIn, 'profiles_u')
        for j = 1:numTotalLines
            profiles_u{j,i} = dataIn.profiles_u{j};
        end
    end
    
    % Store profiles_v if available
    if isfield(dataIn, 'profiles_v')
        for j = 1:numTotalLines
            profiles_v{j,i} = dataIn.profiles_v{j};
        end
    end
    
    % Store X coord in mm for each extraction line
    for j = 1:numTotalLines
        x_mm = dataIn.X(1, dataIn.extractionIndices(j))*1000;
        xCoord_mm(j,i) = x_mm;
    end
end

%% -------------------------------
%  STEP 6: Plot the profiles
% -------------------------------
% Set styling
titleFontSize = 14;
labelFontSize = 12;
tickFontSize  = 10;
lineWidthVal  = 2;

% Colors, markers, lineStyles (user can comment out lineStyles if desired)
colors     = lines(length(fileNames));
markers    = {'o','s','d','^','v','>','<','p','h'};
lineStyles = {'-','--',':','-.'};  % <-- comment out if undesired

% Plot MAIN velocity profiles
label =1;
plotProfilesInFigure(...
    'Comparison of Main Velocity Profiles', ...
    profiles_main, ...
    selectedLineIdx, ...
    xCoord_mm, ...
    throatX_mm, throatY_m, ...
    commonY, ...
    legendEntries, ...
    colors, markers, lineStyles, ...
    titleFontSize, labelFontSize, tickFontSize, lineWidthVal,label);

% Plot profiles_u if any
hasProfilesU = any(cellfun(@(x) ~isempty(x), profiles_u(:)));
label =2;
if hasProfilesU
    plotProfilesInFigure(...
        'Comparison of Average U Velocity Profiles', ...
        profiles_u, ...
        selectedLineIdx, ...
        xCoord_mm, ...
        throatX_mm, throatY_m, ...
        commonY, ...
        legendEntries, ...
        colors, markers, lineStyles, ...
        titleFontSize, labelFontSize, tickFontSize, lineWidthVal,label);
end

% Plot profiles_v if any
hasProfilesV = any(cellfun(@(x) ~isempty(x), profiles_v(:)));
label =3;
if hasProfilesV
    plotProfilesInFigure(...
        'Comparison of Average V Velocity Profiles', ...
        profiles_v, ...
        selectedLineIdx, ...
        xCoord_mm, ...
        throatX_mm, throatY_m, ...
        commonY, ...
        legendEntries, ...
        colors, markers, lineStyles, ...
        titleFontSize, labelFontSize, tickFontSize, lineWidthVal,label);
end

disp('Done!');

%% =========================================================================
%  LOCAL FUNCTIONS (must be at the bottom of the script for R2016b+)
%% =========================================================================

function plotProfilesInFigure(figTitle, profilesCell, ...
    lineIndices, xCoordAll_mm, ...
    throatX_all_mm, throatY_all_m, commonY_m, ...
    legendEntries, colors, markers, lineStyles, ...
    titleFS, labelFS, tickFS, lwVal,label)
% plotProfilesInFigure
% Creates a figure with tiled subplots. Each subplot corresponds to one
% of the user-selected extraction lines. Each line is shifted in the 
% Y-direction according to that file's throat location. The tile title 
% displays the MEAN distance from the throat among all valid files, and 
% the command prompt prints that mean ± std dev.

nFiles   = length(legendEntries);
nLines   = length(lineIndices);

% Create figure
figH = makeProfileFigure(figTitle, nLines);
tl   = makeTileLayout(nLines);
tl.TileIndexing = 'rowmajor';

% We'll store one handle per file for the legend
fileHandles = gobjects(1, nFiles);

for idxTile = 1:nLines
    jLine = lineIndices(idxTile);  % actual extraction line index
    ax    = nexttile;
    hold(ax, 'on');
    
    % ---------------------------------------------------------------
    %  Compute the average distance from throat across all valid files
    %  and also the standard deviation for printing in the command prompt.
    % ---------------------------------------------------------------
    validDiffs = [];  % differences for each file
    for iFile = 1:nFiles
        if ~isempty(profilesCell{jLine, iFile})
            % Distance = throatX_mm(iFile) - xCoord_mm(jLine, iFile)
            thisDiff = throatX_all_mm(iFile)/1000 - xCoordAll_mm(jLine, iFile);
            validDiffs(end+1) = thisDiff; %#ok<AGROW>
        end
    end
    
    if isempty(validDiffs)
        distance_jLine = 0;
        distance_std   = 0;
    else
        distance_jLine = mean(validDiffs, 'omitnan');
        distance_std   = std(validDiffs, 'omitnan');
    end
    
    % Print the distance ± std dev in the command prompt
    fprintf('Line %d: distance = %.3f ± %.3f mm (n=%d files)\n', ...
        jLine, distance_jLine, distance_std, length(validDiffs));
    
    if jLine ==1
        % For the tile title, just show the mean distance
        titleStr = sprintf('Throat');
        title(ax, titleStr, 'FontSize', titleFS);
    else
        % For the tile title, just show the mean distance
        titleStr = sprintf('%.2f mm from throat', abs(distance_jLine));
        title(ax, titleStr, 'FontSize', titleFS);
    end

    if label ==1
        xlabel(ax, 'Velocity (m/s)', 'FontSize', labelFS);
        ylabel(ax, 'Y from throat (mm)', 'FontSize', labelFS);
    elseif label ==2
        xlabel(ax, 'U Velocity (m/s)', 'FontSize', labelFS);
        ylabel(ax, 'Y from throat (mm)', 'FontSize', labelFS);
    else
        xlabel(ax, 'V Velocity (m/s)', 'FontSize', labelFS);
        ylabel(ax, 'Y from throat (mm)', 'FontSize', labelFS);
    end

    set(ax, 'FontSize', tickFS, 'Box','on');

    % ---------------------------------------------------------------
    %  Plot each file's data
    % ---------------------------------------------------------------
    for iFile = 1:nFiles
        profData = profilesCell{jLine, iFile};
        if isempty(profData), continue; end
        
        % SHIFT Y so that the file's throat is at y=0
        yRelative_mm = (commonY_m* 1000 - throatY_all_m(iFile));  % throatY in m

        % Choose color, marker, line style
        colorIdx    = iFile;
        markerIdx   = mod(iFile-1, length(markers)) + 1;
        styleIdx    = mod(iFile-1, length(lineStyles)) + 1;

        % Plot
        hPlot = plot(ax, profData, yRelative_mm, ...
            'Color',     colors(colorIdx,:), ...
            'Marker',    markers{markerIdx}, ...
            'LineStyle', lineStyles{styleIdx}, ...  % comment if you don't want line styles
            'LineWidth', lwVal, ...
            'MarkerSize', 3, ...
            'DisplayName', legendEntries{iFile});

        % For the first tile, store the handle for legend
        if idxTile == 1 && isgraphics(hPlot)
            fileHandles(iFile) = hPlot;
        end
    end

    hold(ax, 'off');
end

% Create legend after all subplots
lgd = legend(fileHandles, legendEntries, 'Location','northwest');
lgd.FontSize  = 8;
lgd.NumColumns = 1;
end

% -------------------------------------------------
function figH = makeProfileFigure(figName, nTiles)
% Creates a figure with a given name and size
figH = figure('Name', figName, ...
    'Position',[100 100 1600 600], ...
    'Color','w');
end

% -------------------------------------------------
function tl = makeTileLayout(nTiles)
% Creates a horizontal tiledlayout with nTiles columns
tl = tiledlayout(1, nTiles, ...
    'TileSpacing','compact', ...
    'Padding','compact');
end
