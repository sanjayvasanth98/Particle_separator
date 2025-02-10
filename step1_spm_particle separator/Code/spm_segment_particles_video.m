function segment_particles_video
    % Prompt user to select the input AVI file
    [filename, pathname] = uigetfile('*.avi', 'Select the video file');
    if isequal(filename,0)
       disp('User selected Cancel');
       return;
    end
    inputFile = fullfile(pathname, filename);

    % Open the video reader
    vReader = VideoReader(inputFile);

    % Prompt user for output location
    [outFilename, outPathname] = uiputfile('*.avi', 'Save processed video as');
    if isequal(outFilename,0)
       disp('User selected Cancel');
       return;
    end
    outputFile = fullfile(outPathname, outFilename);

    % Create a video writer object
    vWriter = VideoWriter(outputFile, 'Uncompressed AVI');
    open(vWriter);

    % Parameters for findMaxima_CAVPTV
    tolerance = 5;          % Adjust based on your image contrast
    strict = false;
    threshold = -Inf;
    lightbackground = false; 
    % 'POINT_SELECTION' gives coordinates of detected centroids
    outputType_points = 'POINT_SELECTION';
    % 'SEGMENTED' provides a watershed-segmented image
    outputType_segmented = 'SEGMENTED';
    excludeOnEdges = false;

    while hasFrame(vReader)
        frame = readFrame(vReader);

        % If frame is colored (RGB), convert to grayscale
        if size(frame, 3) == 3
            grayFrame = rgb2gray(frame);
        else
            grayFrame = frame;
        end

        % STEP 1: Find maxima (centroids) using POINT_SELECTION
        centroids = findMaxima_CAVPTV(grayFrame, tolerance, strict, threshold, lightbackground, outputType_points, excludeOnEdges);
        
        % STEP 2: Get a segmented mask using SEGMENTED mode
        segmentedMask = findMaxima_CAVPTV(grayFrame, tolerance, strict, threshold, lightbackground, outputType_segmented, excludeOnEdges);

        % The segmentedMask returned from watershed segmentation typically uses 255 for particle areas.
        % Make sure we have a binary mask. According to the provided code, 
        % output from 'SEGMENTED' is a watershed result with particle regions = 255.
        particleMask = segmentedMask == 255;

        % Now we have a binary mask of particles. To resemble a PIV image:
        % Set particle pixels to a bright value and the rest to dark (0).
        processedFrame = uint8(particleMask) * 255;

        % If you want to visualize centroids, you could mark them as well (optional):
        % for i = 1:size(centroids,1)
        %     cx = centroids(i,1);
        %     cy = centroids(i,2);
        %     % Ensure coordinates are within the frame
        %     cx = round(cx); cy = round(cy);
        %     if cx > 0 && cx <= size(processedFrame,2) && cy > 0 && cy <= size(processedFrame,1)
        %         processedFrame(cy, cx) = 255; % Mark centroid
        %     end
        % end

        % Convert processedFrame to 3-channel if original was RGB (so it can be written out in color)
        if size(frame,3) == 3
            processedFrame = repmat(processedFrame, [1 1 3]);
        end

        % STEP 3: Write the processed frame to output video
        writeVideo(vWriter, processedFrame);
    end

    % Close the video writer
    close(vWriter);

    disp(['Processed video saved to: ', outputFile]);
end
