    % Prompt user to select the input AVI file
    [filename, pathname] = uigetfile('*.avi', 'Select the video file');
    if isequal(filename,0)
       disp('User selected Cancel');
       return;
    end
    inputFile = fullfile(pathname, filename);

    % Prompt user for output location
    [outFilename, outPathname] = uiputfile('*.avi', 'Save processed video as');
    if isequal(outFilename,0)
       disp('User selected Cancel');
       return;
    end
    outputFile = fullfile(outPathname, outFilename);
   
    %% Enter tolerance value for particle selection
    tolerance = 5;
    strict          = true;
    threshold       = Inf;   % Inf means find all local minima if lightbackground=true
    lightbackground = true;  % true if particles are dark on a bright background
    excludeOnEdges  = true;

    % Enter radius of particle
    radius_of_particle = 1; % radius of particle in pixels
%% Processing step
tic
    spm_process_video(inputFile, outputFile,tolerance,strict,threshold,lightbackground,excludeOnEdges,radius_of_particle)
toc
%%
%% Save the mask on the video for PIV
tic
    spm_ProcessVideoWithMask(inputFile, outputFile)
toc