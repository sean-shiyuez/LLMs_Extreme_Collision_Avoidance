% 清除工作区
clear; clc;

% 加载参数
load('safe_value_params.mat');

% 定义网络架构（移除 regressionLayer）
layers = [
    featureInputLayer(4, 'Name', 'input')  % 输入大小为 4
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(256, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'fc3')  % 输出大小为 1
    % 移除 regressionLayer
];

% 将偏置转换为列向量
bias_fc1 = MLP_0_Dense_0_bias';  % 转置为 [256×1]
bias_fc2 = MLP_0_Dense_1_bias';  % 转置为 [256×1]
bias_fc3 = OutputVDense_bias;    % 已经是标量，无需转换


% 创建 Layer Graph 和 dlnetwork 对象
lgraph = layerGraph(layers);


% 赋值参数
% 修改 fc1 层
fc1Layer = lgraph.Layers(strcmp({lgraph.Layers.Name}, 'fc1'));
fc1Layer.Weights = MLP_0_Dense_0_kernel';
fc1Layer.Bias = bias_fc1;  % 偏置为列向量
lgraph = replaceLayer(lgraph, 'fc1', fc1Layer);

% 修改 fc2 层
fc2Layer = lgraph.Layers(strcmp({lgraph.Layers.Name}, 'fc2'));
fc2Layer.Weights = MLP_0_Dense_1_kernel';
fc2Layer.Bias = bias_fc2;  % 偏置为列向量
lgraph = replaceLayer(lgraph, 'fc2', fc2Layer);

% 修改 fc3 层
fc3Layer = lgraph.Layers(strcmp({lgraph.Layers.Name}, 'fc3'));
fc3Layer.Weights = OutputVDense_kernel';
fc3Layer.Bias = bias_fc3;  % 偏置为标量
lgraph = replaceLayer(lgraph, 'fc3', fc3Layer);

% 创建 dlnetwork 对象
net = dlnetwork(lgraph);


% 测试网络
testInput = rand(4, 1, 'single');
dlTestInput = dlarray(testInput, 'CB');
netOutput = predict(net, dlTestInput);
disp('Network Output:');
disp(netOutput);

% 保存网络
save('safe_value_net.mat', 'net');
