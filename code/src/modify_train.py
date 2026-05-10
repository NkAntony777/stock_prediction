import re

# Read the original train.py
with open('baseline_repo/code/src/train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add AMP imports after the other imports
amp_import = '''

# [Phase 1优化] AMP混合精度训练导入
from torch.cuda.amp import autocast, GradScaler
'''
if 'from torch.cuda.amp import autocast, GradScaler' not in content:
    content = content.replace(
        'import random',
        'import random' + amp_import
    )

# 2. Replace the train_ranking_model function
old_train_func = '''def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # 模型预测
        outputs = model(sequences)  # [batch, max_stocks] 预测分数
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        batch_size = sequences.size(0)
        
        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            
            if valid_indices.numel() == 0:
                continue
                
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
                
            # 获取有效股票的预测值和预处理好的相关性得分
            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]
            
            if len(valid_pred) > 1:
                # 直接使用预处理好的相关性得分，无需重新计算
                loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            if not config.get('drop_clip', True):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                if writer:
                    writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics'''

new_train_func = '''# [Phase 1优化] 排序训练函数 - AMP混合精度和梯度累积
def train_ranking_model(model, dataloader, criterion, optimizer, device, epoch, writer, scaler=None, accumulation_steps=1):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    optimizer.zero_grad()
    accumulation_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        # [Phase 1优化] AMP混合精度前向传播
        with autocast(enabled=config.get('use_amp', True)):
            # 模型预测
            outputs = model(sequences)  # [batch, max_stocks] 预测分数
            
            # 应用mask，只考虑有效股票
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
            masked_targets = targets * masks
            masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
            
            # 计算损失（只对有效股票计算）
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                    
                # 获取有效股票的预测值和预处理好的相关性得分
                valid_pred = masked_outputs[i][valid_indices]
                valid_relevance = masked_relevance[i][valid_indices]
                
                if len(valid_pred) > 1:
                    # 直接使用预处理好的相关性得分，无需重新计算
                    loss = criterion(valid_pred.unsqueeze(0), valid_relevance.unsqueeze(0))
                    batch_loss = batch_loss + loss if isinstance(batch_loss, torch.Tensor) else loss
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                # [Phase 1优化] 梯度累积：除以accumulation_steps
                accumulation_loss += batch_loss / accumulation_steps
        
        # [Phase 1优化] 梯度累积逻辑
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.scale(accumulation_loss).backward()
                
                if config.get('drop_clip', True):
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                accumulation_loss.backward()
                if config.get('drop_clip', True):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    if writer:
                        writer.add_scalar('train/grad_norm', grad_norm, global_step=epoch*len(dataloader)+local_step)
                optimizer.step()
            
            optimizer.zero_grad()
            
            total_loss += accumulation_loss.item() * accumulation_steps
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            accumulation_loss = 0
            
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics'''

content = content.replace(old_train_func, new_train_func)

# 3. Modify the main() function to add scaler initialization and pass to train_ranking_model
old_main_part = '''    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0)
    )  # 使用加权排序损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs'])'''

new_main_part = '''    # 7. 损失函数和优化器
    criterion = WeightedRankingLoss(
        k=5,
        temperature=1.0,
        weight_factor=config['top5_weight'],
        pairwise_weight=config['pairwise_weight'],
        base_weight=config.get('base_weight', 1.0)
    )  # 使用加权排序损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.2, total_iters=config['num_epochs'])
    
    # [Phase 1优化] 初始化AMP GradScaler
    scaler = GradScaler() if config.get('use_amp', True) and device.type == 'cuda' else None
    accumulation_steps = config.get('accumulation_steps', 1)'''

content = content.replace(old_main_part, new_main_part)

# 4. Modify the train_ranking_model call to pass scaler and accumulation_steps
old_train_call = '''            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer
            )'''

new_train_call = '''            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                scaler=scaler, accumulation_steps=accumulation_steps
            )'''

content = content.replace(old_train_call, new_train_call)

# Write the modified content back
with open('baseline_repo/code/src/train.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('train.py modified successfully!')
