function digit_project
clc; close all;

%% LOAD OR TRAIN MODEL
if isfile('trainedModel.mat')
    load('trainedModel.mat','net');
else
    load('mnist.mat','training');

    XTrain = reshape(training.images,28,28,1,[]);
    YTrain = categorical(training.labels);

    XTrain = XTrain(:,:,:,1:10000);
    YTrain = YTrain(1:10000);

    layers = [
        imageInputLayer([28 28 1])
        convolution2dLayer(3,8,'Padding','same')
        reluLayer
        maxPooling2dLayer(2,'Stride',2)

        convolution2dLayer(3,16,'Padding','same')
        reluLayer
        maxPooling2dLayer(2,'Stride',2)

        fullyConnectedLayer(64)
        reluLayer

        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];

    options = trainingOptions('adam','MaxEpochs',5,'Verbose',false);

    net = trainNetwork(XTrain,YTrain,layers,options);
    save('trainedModel.mat','net');
end

%% TEST ACCURACY
load('mnist.mat','test');
XTest = reshape(test.images,28,28,1,[]);
YTest = categorical(test.labels);
YPred = classify(net, XTest);
model_accuracy = sum(YPred == YTest) / numel(YTest) * 100;

%% USER CHOICE
disp('1. Predict from Image');
disp('2. Draw & Predict (GUI)');
choice = input('Enter choice: ');

if choice==1
    predict_image(net, model_accuracy);
else
    draw_gui(net, model_accuracy);
end
end

%% IMAGE MODE
function predict_image(net, model_accuracy)
[file,path] = uigetfile({'*.png;*.jpg'});
if isequal(file,0), return; end

original = imread(fullfile(path,file));
img = original;
if size(img,3)==3, img = rgb2gray(img); end
img = im2double(img);

bw = imbinarize(img);
if mean(bw(:)) > 0.5, bw = ~bw; end

bw = imfill(bw,'holes');
bw = imclose(bw, strel('disk',2));
bw = bwareaopen(bw,100);

predict_digits(bw, original, net, model_accuracy);
end

%% DRAW MODE
function draw_gui(net, model_accuracy)

fig = figure('Position',[300 100 600 650]);

canvas_size = 500;
canvas = axes('Parent',fig,'Units','pixels',...
    'Position',[50 120 canvas_size canvas_size]);

drawing = zeros(canvas_size);
imshow(drawing,'Parent',canvas);

isDrawing = false; prevPoint = [];

set(fig,'WindowButtonDownFcn',@startDraw);
set(fig,'WindowButtonUpFcn',@stopDraw);
set(fig,'WindowButtonMotionFcn',@draw);

uicontrol('Style','pushbutton','String','Predict',...
    'Position',[150 30 100 50],'Callback',@predict);

uicontrol('Style','pushbutton','String','Clear',...
    'Position',[300 30 100 50],'Callback',@clearCanvas);

function startDraw(~,~), isDrawing=true; prevPoint=getPoint(); end
function stopDraw(~,~), isDrawing=false; prevPoint=[]; end
function pt=getPoint(), cp=get(canvas,'CurrentPoint'); pt=round(cp(1,1:2)); end

function draw(~,~)
if isDrawing
    currPoint=getPoint();
    if ~isempty(prevPoint)
        x=linspace(prevPoint(1),currPoint(1),30);
        y=linspace(prevPoint(2),currPoint(2),30);
        for k=1:length(x)
            xi=round(x(k)); yi=round(y(k));
            if xi>4 && yi>4 && xi<canvas_size-4 && yi<canvas_size-4
                drawing(yi-4:yi+4,xi-4:xi+4)=1;
            end
        end
    end
    prevPoint=currPoint;
    imshow(drawing,'Parent',canvas);
end
end

function clearCanvas(~,~)
drawing(:)=0;
imshow(drawing,'Parent',canvas);
end

function predict(~,~)
bw = imbinarize(drawing);
bw = imfill(bw,'holes');
bw = imopen(bw, strel('disk',1));
bw = imclose(bw, strel('disk',2));
bw = bwareaopen(bw,200);

predict_digits(bw, drawing, net, model_accuracy);
end
end

%% MAIN PREDICTION
function predict_digits(bw, original, net, model_accuracy)

stats = regionprops(bw,'BoundingBox','Area');
stats = stats([stats.Area] > 200);

boxes = reshape([stats.BoundingBox],4,[])';
[~,order] = sort(boxes(:,1));
boxes = boxes(order,:);

num_digits = size(boxes,1);
conf_list = zeros(1,num_digits);
final_number = strings(1,num_digits);

all_digits = cell(1,num_digits);

for idx = 1:num_digits
    digit_img = imcrop(bw, boxes(idx,:));

    props = regionprops(digit_img,'BoundingBox');
    digit_img = imcrop(digit_img, props(1).BoundingBox);

    [h,w]=size(digit_img);
    size_max=max(h,w);
    square_img=zeros(size_max);

    y_offset=floor((size_max-h)/2);
    x_offset=floor((size_max-w)/2);

    square_img(y_offset+1:y_offset+h,...
               x_offset+1:x_offset+w)=digit_img;

    digit_img=imresize(square_img,[28 28]);

    all_digits{idx} = digit_img;

    figure;
    subplot(1,2,1), imshow(original), title('Original Input')
    subplot(1,2,2), imshow(digit_img), title('Processed (Model Input)')

    input_img=reshape(digit_img,28,28,1);

    [label,scores]=classify(net,input_img);
    predicted_digit = double(label)-1;

    conf_list(idx)=max(scores)*100;
    final_number(idx)=string(predicted_digit);
end

final_number = char(join(final_number,""));
avg_conf = mean(conf_list);

%% RESULT WINDOW
figure;
imshow(original)
title(['Predicted: ', final_number],'FontSize',14,'Color','blue')

%% BUTTONS
uicontrol('Style','pushbutton','String','✔ Correct',...
    'Position',[100 20 100 40],'Callback',@correct);

uicontrol('Style','pushbutton','String','✖ Wrong',...
    'Position',[250 20 100 40],'Callback',@wrong);

function correct(~,~)
    show_graph();
end

function wrong(~,~)
    answer = inputdlg({'Enter Predicted Digit:','Enter Correct Digit:'});
    if isempty(answer) || isempty(answer{1}) || isempty(answer{2}), return; end

    title(['Predicted: ', answer{1},' | Correct: ', answer{2}],'Color','red')

   %% ✅ FIXED HEATMAP WITH TEXT (multi-digit)

for d = 1:num_digits

    figure;

    % crop SAME digit from original
    orig_digit = imcrop(bw, boxes(d,:));
    orig_small = imresize(orig_digit,[28 28]);
    orig_small = im2double(orig_small);

    digit_img = all_digits{d};

    diff_img = abs(orig_small - digit_img);

    subplot(1,3,1)
    imshow(orig_small)
    title('Original Digit')

    subplot(1,3,2)
    imshow(digit_img)
    title(['Processed Digit ', num2str(d)])

    subplot(1,3,3)
    imagesc(diff_img)
    colormap jet
    colorbar
    title(['Heatmap Digit ', num2str(d)])

    %% 🔥 TEXT (clean, no overlap)

    if avg_conf < 60
        feedback = 'Low confidence → unclear drawing';
    elseif avg_conf < 85
        feedback = 'Medium confidence → improve shape';
    else
        feedback = 'High confidence → good drawing';
    end

    msg = sprintf(['Digit ', ' Analysis:\n',...
                   'Red/Yellow = Model confused\n',...
                   'Blue = Good match\n',...
                   'Feedback: ', feedback]);

    sgtitle(msg,'FontSize',11,'FontWeight','bold','Color','white');

end
    show_graph();
end

    function show_graph()

    figure;

    values = [avg_conf, model_accuracy];

    b = bar(values,0.4);
    b.FaceColor='flat';
    b.CData(1,:)=[0.2 0.6 1];   % Confidence
    b.CData(2,:)=[0.2 0.8 0.4]; % Accuracy

    ylim([0 110])
    set(gca,'XTickLabel',{'Confidence (%)','Model Accuracy (%)'})
    title('Model Performance')
    grid on

    for k=1:length(values)
        text(k, values(k)+3, sprintf('%.1f%%',values(k)),...
            'HorizontalAlignment','center');
    end
end
end