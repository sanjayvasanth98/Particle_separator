function spm_process_video(inputFile, outputFile,tolerance,strict,threshold,lightbackground,excludeOnEdges,radius_of_particle)
    % Adjust these parameters as needed:
    tolerance       = tolerance;     % Try smaller values to detect more maxima
    strict          = strict;
    threshold       = threshold;   % Inf means find all local minima if lightbackground=true
    lightbackground = lightbackground;  % true if particles are dark on a bright background
    excludeOnEdges  = excludeOnEdges;

    % Open the video reader
    vReader = VideoReader(inputFile);
    
    % Create a video writer
    vWriter = VideoWriter(outputFile, 'Motion JPEG AVI');
    open(vWriter);

    processedFrames = {};
    frameCount = 0;
    
    % Read and process each frame
    while hasFrame(vReader)
        frameCount = frameCount + 1;
        frame = readFrame(vReader);

        % Convert to grayscale if needed
        if size(frame, 3) == 3
            grayFrame = rgb2gray(frame);
            isColor = true;
        else
            grayFrame = frame;
            isColor   = false;
        end

        % STEP 1: Detect Particle Centroids
        centroids = findMaxima_CAVPTV(grayFrame, tolerance, strict, threshold, ...
            lightbackground, 'POINT_SELECTION', excludeOnEdges);
        fprintf('Frame %d: Detected %d particles.\n', frameCount, size(centroids,1));

        % If no particles, just output a black frame
        if isempty(centroids)
            processedFrame = zeros(size(grayFrame), 'uint8');
            if isColor
                processedFrame = repmat(processedFrame, [1 1 3]);
            end
        else
            % STEP 2: Create initial binary mask of particles
            % Use Otsu thresholding for a start:
            level = graythresh(grayFrame);
            bw = imbinarize(grayFrame, level);

            % Dilate the binary mask slightly to make them bigger
            se = strel('disk', 1);
            bw = imdilate(bw, se);

            % Select only the areas around the detected centroids
            radiusROI = radius_of_particle; % radius for initial ROI %only radius =1 works
            particleMask = false(size(bw));
            for i = 1:size(centroids,1)
                cx = round(centroids(i,1));
                cy = round(centroids(i,2));
                xRange = max(1, cx - radiusROI):min(size(bw,2), cx + radiusROI);
                yRange = max(1, cy - radiusROI):min(size(bw,1), cy + radiusROI);
                particleMask(yRange, xRange) = particleMask(yRange, xRange) | bw(yRange, xRange);
            end
            
            % STEP 3: For each detected particle, fit a circle
            CC = bwconncomp(particleMask);
            circularMask = false(size(particleMask));
            
            for i = 1:CC.NumObjects
                pixelIdxList = CC.PixelIdxList{i};
                [yPix, xPix] = ind2sub(size(particleMask), pixelIdxList);
                
                % Compute centroid
                xC = mean(xPix);
                yC = mean(yPix);
                
                % Compute area of particle
                areaParticle = length(pixelIdxList);
                
                % Compute radius to match the particle area with a circle
                radius = sqrt(areaParticle/pi);
                
                % Create a circular mask for this particle
                [X, Y] = meshgrid(1:size(particleMask,2), 1:size(particleMask,1));
                distFromCentroid = sqrt((X - xC).^2 + (Y - yC).^2);
                particleCircle = distFromCentroid <= radius;
                
                % Combine this circle with the main mask
                circularMask = circularMask | particleCircle;
            end
            
            % STEP 4: Use the circular mask to segment the original image
            % Set background to black, particles retain their original intensity
            segmentedParticles = zeros(size(grayFrame), 'uint8');
            segmentedParticles(circularMask) = grayFrame(circularMask);

            % Convert segmented particles to color if needed
            if isColor
                processedFrame = repmat(segmentedParticles, [1 1 3]);
            else
                processedFrame = segmentedParticles;
            end

            % **Show outline on the original image for the first frame only** 
            if frameCount == 1
                % -- Create an overlay of the circle boundaries on the original frame --
                % Get perimeter of circularMask
                boundaryMask = bwperim(circularMask);
                
                % If the original was color, overlay on it directly;
                % otherwise, convert grayscale to 3-channels for display
                if ~isColor
                    overlayFrame = repmat(frame, [1 1 3]); % Gray->3-ch
                else
                    overlayFrame = frame;
                end

                % Mark the perimeter in red
                redColor = uint8([255, 0, 0]); 
                for c = 1:3
                    channel = overlayFrame(:,:,c);
                    channel(boundaryMask) = redColor(c);
                    overlayFrame(:,:,c) = channel;
                end

                % Display the result
                figure('Name','Verify Particles in Frame 1','NumberTitle','off');
                imshow(overlayFrame);
                title('Particle outlines in red on original image (Frame 1)');
                
                % Ask user to confirm
                choice = questdlg('Does this outline look correct?', ...
                    'Confirmation','Yes','No','Yes');
                if strcmp(choice, 'No')
                    disp('User did not confirm. Stopping.');
                    % Close the writer, clean up, and return
                    close(vWriter);
                    return
                else
                    disp('User confirmed. Proceeding with the rest of the video.');
                end
            end
        end

        % Write processed frame to output
        writeVideo(vWriter, processedFrame);
        processedFrames{frameCount} = processedFrame;
    end

    % Close the video writer
    close(vWriter);

    % Display frames with a slider
    display_frames_with_slider(processedFrames);
end


function display_frames_with_slider(frames)
    numFrames = length(frames);
    hFig = figure('Name','Processed Frames Viewer','NumberTitle','off');
    hAx = axes('Parent', hFig, 'Units','normalized', 'Position',[0.05 0.15 0.9 0.8]);

    imshow(frames{1}, 'Parent', hAx);
    title(hAx, sprintf('Frame 1 of %d', numFrames));

    hSlider = uicontrol('Parent', hFig, 'Style', 'slider', ...
        'Units','normalized', 'Position',[0.05 0.05 0.9 0.05], ...
        'Value',1, 'Min',1, 'Max',numFrames, ...
        'SliderStep',[1/(numFrames-1), 1/(numFrames-1)]);

    addlistener(hSlider, 'Value', 'PostSet', @(src,evt) updateFrame());

    function updateFrame()
        frameIdx = round(hSlider.Value);
        imshow(frames{frameIdx}, 'Parent', hAx);
        title(hAx, sprintf('Frame %d of %d', frameIdx, numFrames));
    end
end
