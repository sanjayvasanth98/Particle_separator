function spm_ProcessVideoWithMask(inputFile, outputFile)
    % spm_ProcessVideoWithMask Reads a video from inputFile, allows the user
    % to interactively create a mask, overlays the mask on the video loaded 
    % from outputFile, and saves the result with a unique name.
    %
    % Inputs:
    % - inputFile: String containing the path to the input .avi video for mask creation.
    % - outputFile: String containing the path to the .avi video where the mask is applied.
    
    % Load input video for creating the mask
    vidObjInput = VideoReader(inputFile);
    
    % Read the first frame to create the mask
    firstFrame = readFrame(vidObjInput);
    
    % Display the first frame and allow the user to create a mask
    mask = CreateMask_CAVPTV(firstFrame);
    
    % Confirm with the user before proceeding
    confirmMsg = 'Do you want to overlay the selected mask on the output video? (y/n): ';
    userResponse = input(confirmMsg, 's');
    if lower(userResponse) ~= 'y'
        disp('Mask overlay canceled by the user.');
        return;
    end
    
    % Load the output video for applying the mask
    vidObjOutput = VideoReader(outputFile);
    
    % Generate a unique name for the processed video
    [path, name, ~] = fileparts(outputFile);
    timestamp = datestr(now, 'yyyymmdd_HHMMSS'); % Unique timestamp
    processedVideoPath = fullfile(path, [name '_masked_' timestamp '.avi']);
    
    % Create a VideoWriter object for the processed video
    outVidObj = VideoWriter(processedVideoPath);
    outVidObj.FrameRate = vidObjOutput.FrameRate;
    open(outVidObj);
    
    % Reset video reader to the beginning
    vidObjOutput.CurrentTime = 0;
    
    % Process each frame from the output video and overlay the mask
    while hasFrame(vidObjOutput)
        frame = readFrame(vidObjOutput);
        
        % Convert mask to RGB (white outline on the frame)
        maskOutline = bwperim(mask); % Get the mask outline
        frame(maskOutline) = 255;    % Set outline to white in grayscale frame
        
        % Write the processed frame to the new video
        writeVideo(outVidObj, frame);
    end
    
    % Close the video writer
    close(outVidObj);
    
    % Save the mask as a .mat file
    maskFilePath = fullfile(path, [name '_mask_' timestamp '.mat']);
    save(maskFilePath, 'mask');
    
    % Display completion message
    disp(['Video processing completed. Output saved to: ' processedVideoPath]);
    disp(['Mask saved to: ' maskFilePath]);
end
