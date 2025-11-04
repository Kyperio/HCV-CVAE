"""
Safe KL Annealing Strategies
Provides numerically stable KL annealing with safety checks and adaptive mechanisms.
"""

import torch
import math
from typing import Optional


class SafeKLAnnealing:
    """Safe KL annealing with numerical stability checks"""
    
    def __init__(self, strategy='linear', max_weight=0.001, anneal_epochs=100, 
                 min_weight=1e-6, kl_target_min=0.1, kl_target_max=10.0):
        self.strategy = strategy
        self.max_weight = max_weight
        self.anneal_epochs = anneal_epochs
        self.min_weight = min_weight
        self.kl_target_min = kl_target_min
        self.kl_target_max = kl_target_max
        
        # 安全参数
        self.kl_div_history = []
        self.weight_history = []
        self.nan_count = 0
        self.max_nan_count = 10
        
    def get_weight(self, epoch: int, kl_div: Optional[torch.Tensor] = None) -> float:
        """获取安全的KL权重"""
        
        # 检查KL散度是否有效
        if kl_div is not None:
            kl_div_value = kl_div.item() if torch.is_tensor(kl_div) else kl_div
            
            # 检查数值稳定性
            if torch.isnan(kl_div) or torch.isinf(kl_div) or kl_div_value < 0:
                self.nan_count += 1
                print(f"⚠️  Invalid KL divergence: {kl_div_value}, count: {self.nan_count}")
                
                if self.nan_count > self.max_nan_count:
                    print("🚨  Too many invalid KL divergences, using emergency fallback")
                    return self.min_weight
                
                # 使用历史平均值或默认值
                if self.kl_div_history:
                    kl_div_value = sum(self.kl_div_history[-10:]) / min(len(self.kl_div_history), 10)
                else:
                    kl_div_value = 1.0
            
            self.kl_div_history.append(kl_div_value)
            
            # 限制历史长度
            if len(self.kl_div_history) > 100:
                self.kl_div_history = self.kl_div_history[-50:]
        
        # 根据策略计算权重
        if self.strategy == 'linear':
            weight = self._linear_annealing(epoch)
        elif self.strategy == 'cosine':
            weight = self._cosine_annealing(epoch)
        elif self.strategy == 'adaptive':
            weight = self._adaptive_annealing(epoch, kl_div_value if kl_div is not None else 1.0)
        elif self.strategy == 'safe_linear':
            weight = self._safe_linear_annealing(epoch)
        else:
            weight = self._linear_annealing(epoch)
        
        # 应用安全限制
        weight = max(self.min_weight, min(weight, self.max_weight))
        
        # 检查权重是否有效
        if math.isnan(weight) or math.isinf(weight):
            print(f"⚠️  Invalid weight calculated: {weight}, using fallback")
            weight = self.min_weight
        
        self.weight_history.append(weight)
        return weight
    
    def _linear_annealing(self, epoch: int) -> float:
        """线性退火"""
        if epoch <= self.anneal_epochs:
            return self.max_weight * (epoch / self.anneal_epochs)
        else:
            return self.max_weight
    
    def _cosine_annealing(self, epoch: int) -> float:
        """余弦退火"""
        if epoch <= self.anneal_epochs:
            return self.max_weight * (1 + math.cos(math.pi * epoch / self.anneal_epochs)) / 2
        else:
            return self.max_weight
    
    def _adaptive_annealing(self, epoch: int, kl_div_value: float) -> float:
        """自适应退火"""
        base_weight = self.max_weight * (epoch / self.anneal_epochs) if epoch <= self.anneal_epochs else self.max_weight
        
        # 根据KL散度调整权重
        if kl_div_value < self.kl_target_min:
            # KL散度太小，增加权重
            adjustment = 1.2
        elif kl_div_value > self.kl_target_max:
            # KL散度太大，减少权重
            adjustment = 0.8
        else:
            # 正常范围
            adjustment = 1.0
        
        return base_weight * adjustment
    
    def _safe_linear_annealing(self, epoch: int) -> float:
        """安全的线性退火，包含额外的稳定性检查"""
        if epoch <= self.anneal_epochs:
            progress = epoch / self.anneal_epochs
            # 使用更平滑的增长曲线
            weight = self.max_weight * (progress ** 0.5)  # 平方根增长
        else:
            weight = self.max_weight
        
        return weight
    
    def get_statistics(self) -> dict:
        """获取退火统计信息"""
        if not self.kl_div_history:
            return {"message": "No history available"}
        
        return {
            "kl_div_mean": sum(self.kl_div_history) / len(self.kl_div_history),
            "kl_div_std": math.sqrt(sum((x - sum(self.kl_div_history) / len(self.kl_div_history)) ** 2 for x in self.kl_div_history) / len(self.kl_div_history)),
            "kl_div_min": min(self.kl_div_history),
            "kl_div_max": max(self.kl_div_history),
            "nan_count": self.nan_count,
            "weight_mean": sum(self.weight_history) / len(self.weight_history) if self.weight_history else 0
        }


def get_safe_kl_annealing(strategy='safe_linear', **kwargs):
    """获取安全的KL退火器"""
    return SafeKLAnnealing(strategy=strategy, **kwargs)


# 预定义的安全配置
SAFE_CONFIGS = {
    'conservative': {
        'strategy': 'safe_linear',
        'max_weight': 0.0005,
        'anneal_epochs': 150,
        'min_weight': 1e-7
    },
    'moderate': {
        'strategy': 'linear',
        'max_weight': 0.001,
        'anneal_epochs': 100,
        'min_weight': 1e-6
    },
    'adaptive': {
        'strategy': 'adaptive',
        'max_weight': 0.002,
        'anneal_epochs': 80,
        'min_weight': 1e-6,
        'kl_target_min': 0.5,
        'kl_target_max': 5.0
    }
}


def get_predefined_safe_annealing(config_name='conservative'):
    """获取预定义的安全退火配置"""
    if config_name not in SAFE_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(SAFE_CONFIGS.keys())}")
    
    return SafeKLAnnealing(**SAFE_CONFIGS[config_name])


if __name__ == "__main__":
    # 测试安全退火
    print("=== Testing Safe KL Annealing ===")
    
    # 测试正常情况
    annealer = get_safe_kl_annealing('safe_linear', max_weight=0.001, anneal_epochs=50)
    
    for epoch in range(60):
        kl_div = torch.tensor(1.0 + 0.5 * torch.sin(epoch * 0.1))
        weight = annealer.get_weight(epoch, kl_div)
        print(f"Epoch {epoch:2d}: KL div = {kl_div.item():.4f}, Weight = {weight:.6f}")
    
    # 测试异常情况
    print("\n=== Testing with Invalid KL Divergence ===")
    annealer2 = get_safe_kl_annealing('adaptive', max_weight=0.002, anneal_epochs=30)
    
    for epoch in range(10):
        if epoch == 5:
            kl_div = torch.tensor(float('nan'))  # 模拟NaN
        else:
            kl_div = torch.tensor(1.0)
        
        weight = annealer2.get_weight(epoch, kl_div)
        print(f"Epoch {epoch:2d}: KL div = {kl_div.item() if not torch.isnan(kl_div) else 'NaN'}, Weight = {weight:.6f}")
    
    # 显示统计信息
    print(f"\n=== Statistics ===")
    stats = annealer.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")