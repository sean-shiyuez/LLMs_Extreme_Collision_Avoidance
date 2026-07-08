import pickle
import numpy as np
import torch
import jax
import jax.numpy as jnp
import scipy.io

# 读取模型
with open('model314.pickle', 'rb') as f:
    model = pickle.load(f)

print("Type of model['safe_value']:", type(model['safe_value']))
print("Keys in model['safe_value']:", model['safe_value'].keys())

params = model['safe_value']['params']  # 获取参数

# 统一转换为 NumPy 格式
def params_to_numpy(params):
    return jax.tree_util.tree_map(lambda x: np.array(x), params)

numpy_params = params_to_numpy(params)

# 平展参数字典
def flatten_params(params, parent_key='', sep='_'):
    items = []
    for k, v in params.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_params(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

flattened_params = flatten_params(numpy_params)

# 保存为 MATLAB .mat 格式
scipy.io.savemat('safe_value_params.mat', flattened_params)

# 保存 NumPy 格式
np.savez('safe_value_params.npz', **flattened_params)

# 如果想用 PyTorch 兼容
torch_params = {k: torch.tensor(v) for k, v in flattened_params.items()}
torch.save(torch_params, 'safe_value_params.pth')

print("Model parameters saved as .mat, .npz, and .pth!")

# 生成加载代码
with open("load_model.py", "w") as f:
    f.write('''import torch
import numpy as np
import scipy.io

def load_params(format='npz'):
    if format == 'mat':
        return scipy.io.loadmat('safe_value_params.mat')
    elif format == 'npz':
        return np.load('safe_value_params.npz', allow_pickle=True)
    elif format == 'pth':
        return torch.load('safe_value_params.pth')
    else:
        raise ValueError("Unsupported format! Use 'mat', 'npz', or 'pth'.")

print("Model parameters loaded!")
''')

print("Python model loader script 'load_model.py' generated!")
