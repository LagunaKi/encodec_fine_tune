# EnCodec微调项目 - 音频片段相对复杂度预测

本项目基于EnCodec模型进行微调，用于预测相邻音频片段之间的相对复杂度差异（使用信息量IC指标）。

## 功能特点

- 预测相邻音频片段间的复杂度差异
- 使用KL散度计算信息量直方图差异作为特征
- 支持PyTorch DDP分布式训练
- 支持可变长度音频片段（带padding掩码）

## 系统要求

- Python 3.8+
- PyTorch 1.12+
- EnCodec模型
- torchaudio
- scikit-learn
- NVIDIA显卡驱动（建议CUDA 11.7+）

## 数据集准备

1. 将音频文件放入`dataset/verified_mp3/`目录
2. 准备标签文件`dataset/labels.csv`，需包含以下列：
   - `segment_id`: 片段ID
   - `start_time`: 开始时间(秒)
   - `end_time`: 结束时间(秒) 
   - `complexity`: 复杂度分数

## 训练示例（2张RTX 3060）
新建会话：
``` bash
screen -S train_session
```
返回会话：
``` bash
screen -D -r train_session
```
查看会话列表：
``` bash
screen -ls 
```
检查当前会话：
``` bash
echo $STY
```

```bash
PYTHONPATH=. python -m torch.distributed.run --nproc_per_node=2 ./fine_tune/train.py \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0003 \
  --save_dir ./checkpoints
```

单gpu:
PYTHONPATH=. python ./fine_tune/train.py \
  --batch_size 8 \
  --epochs 50 \
  --lr 0.0003 \
  --save_dir ./checkpoints

## 模型架构

### 新架构特点
- 基于TCN(时间卷积网络)的时序特征提取
- 交叉注意力机制比较变奏差异
- 不再依赖IC特征
- 更专注于片段间差异建模

### 详细结构
1. **特征提取层**
   - 预训练EnCodec编码器（冻结参数）
   - 4层TCN网络（64通道）
   - 3x3卷积核，保持时序长度

2. **差异比较层**
   - 4头交叉注意力机制
   - 当前片段作为query
   - 前一片段作为key/value
   - 自动学习重要差异特征

3. **预测头**
   - 结合特征差值和注意力输出
   - 256维隐藏层
   - 直接预测复杂度差值

### 优势
- 更准确捕捉变奏特征变化
- 注意力权重提供可解释性
- 端到端训练更稳定

## 预测使用

```python
from fine_tune.model_ext import RelativeComplexityModel

# 加载训练好的模型
model = RelativeComplexityModel()
model.load_state_dict(torch.load("checkpoints/best_model.pth"))

# prev_seg: 前一片段, (1, 1, 样本数)的Tensor
# curr_seg: 当前片段, (1, 1, 样本数)的Tensor 
delta = model.predict_complexity(prev_seg, curr_seg)
```


### 远程GPU服务器训练指南

1. **连接服务器**
```bash
ssh fd-lamt-04@10.177.64.182
# 输入密码后登录
```

