# 配置参数 - V3 优化版
# 核心改动：恢复d_model=256，提升模型容量
# V2成功经验复用：learning_rate=3e-5, top5_weight=3.5, CosineAnnealing

sequence_length = 60
feature_num = '158+39'
config = {
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据
    'd_model': 256,         # [核心] 恢复256，大幅提升模型容量
    'nhead': 8,             # [核心] 恢复8头注意力 (256/32=8)
    'num_layers': 4,        # [核心] 恢复4层Transformer
    'dim_feedforward': 1024, # [核心] 恢复1024前馈层
    'batch_size': 1,         # 降到1避免OOM，配合accumulation_steps使用
    'num_epochs': 80,       # 训练轮数
    'learning_rate': 3e-5,  # [V2成功经验] 使用3e-5

    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    # 损失函数权重 - [V2成功经验] top5_weight=3.5
    'pairwise_weight': 2,
    'base_weight': 0.5,
    'top5_weight': 3.5,
    'top10_weight': 2.5,

    'output_dir': f'./model/{sequence_length}_{feature_num}_v3',
    'data_path': './data',
    'use_selected_features': False,

    # AMP混合精度训练配置
    'use_amp': True,
    'accumulation_steps': 16,  # 配合batch_size=1，累积16步等效batch=16

    # 学习率调度器配置 - [V2成功经验] CosineAnnealing
    'scheduler_type': 'cosine',
    'warmup_epochs': 5,

    # 正则化配置
    'weight_decay': 2e-5,
    'label_smoothing': 0.0,
}