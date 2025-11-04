"""
KL Annealing Strategies for CVAE Training
Provides various KL annealing strategies to improve training stability and generation quality.
"""

import math
import torch


class KLAnnealing:
    """Base class for KL annealing strategies"""
    
    def __init__(self, max_weight=0.01, **kwargs):
        self.max_weight = max_weight
        
    def get_weight(self, epoch, kl_div=None):
        """Get KL weight for current epoch"""
        raise NotImplementedError


class LinearAnnealing(KLAnnealing):
    """Linear KL annealing strategy"""
    
    def __init__(self, anneal_epochs=50, max_weight=0.01):
        super().__init__(max_weight)
        self.anneal_epochs = anneal_epochs
        
    def get_weight(self, epoch, kl_div=None):
        if epoch <= self.anneal_epochs:
            return self.max_weight * (epoch / self.anneal_epochs)
        else:
            return self.max_weight


class CyclicalAnnealing(KLAnnealing):
    """Cyclical KL annealing strategy"""
    
    def __init__(self, cycle_length=20, max_weight=0.01):
        super().__init__(max_weight)
        self.cycle_length = cycle_length
        
    def get_weight(self, epoch, kl_div=None):
        cycle_position = (epoch % self.cycle_length) / self.cycle_length
        if cycle_position < 0.5:
            return self.max_weight * (2 * cycle_position)  # Ramp up
        else:
            return self.max_weight * (2 * (1 - cycle_position))  # Ramp down


class CosineAnnealing(KLAnnealing):
    """Cosine KL annealing strategy"""
    
    def __init__(self, anneal_epochs=50, max_weight=0.01):
        super().__init__(max_weight)
        self.anneal_epochs = anneal_epochs
        
    def get_weight(self, epoch, kl_div=None):
        if epoch <= self.anneal_epochs:
            return self.max_weight * (1 + math.cos(math.pi * epoch / self.anneal_epochs)) / 2
        else:
            return self.max_weight


class AdaptiveAnnealing(KLAnnealing):
    """Adaptive KL annealing based on KL divergence value"""
    
    def __init__(self, base_weight=0.01, min_weight=0.001, max_weight=0.1, 
                 kl_target_min=10, kl_target_max=100):
        super().__init__(max_weight)
        self.base_weight = base_weight
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.kl_target_min = kl_target_min
        self.kl_target_max = kl_target_max
        self.current_weight = base_weight
        
    def get_weight(self, epoch, kl_div=None):
        if kl_div is not None:
            kl_div_value = kl_div.item() if torch.is_tensor(kl_div) else kl_div
            
            if kl_div_value < self.kl_target_min:  # KL too small, increase weight
                self.current_weight = min(self.max_weight, self.current_weight * 1.1)
            elif kl_div_value > self.kl_target_max:  # KL too large, decrease weight
                self.current_weight = max(self.min_weight, self.current_weight * 0.9)
                
        return self.current_weight


class BetaAnnealing(KLAnnealing):
    """Beta-VAE style annealing with warmup"""
    
    def __init__(self, warmup_epochs=20, max_weight=0.01, beta_schedule='linear'):
        super().__init__(max_weight)
        self.warmup_epochs = warmup_epochs
        self.beta_schedule = beta_schedule
        
    def get_weight(self, epoch, kl_div=None):
        if epoch < self.warmup_epochs:
            if self.beta_schedule == 'linear':
                return self.max_weight * (epoch / self.warmup_epochs)
            elif self.beta_schedule == 'cosine':
                return self.max_weight * (1 + math.cos(math.pi * (1 - epoch / self.warmup_epochs))) / 2
        else:
            return self.max_weight


def get_kl_annealing(strategy='linear', **kwargs):
    """Factory function to get KL annealing strategy"""
    
    strategies = {
        'linear': LinearAnnealing,
        'cyclical': CyclicalAnnealing,
        'cosine': CosineAnnealing,
        'adaptive': AdaptiveAnnealing,
        'beta': BetaAnnealing
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown KL annealing strategy: {strategy}. "
                        f"Available strategies: {list(strategies.keys())}")
    
    return strategies[strategy](**kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test different annealing strategies
    strategies = ['linear', 'cyclical', 'cosine', 'adaptive', 'beta']
    
    for strategy_name in strategies:
        print(f"\n{strategy_name.upper()} Annealing:")
        annealer = get_kl_annealing(strategy_name)
        
        for epoch in range(0, 60, 10):
            weight = annealer.get_weight(epoch)
            print(f"Epoch {epoch:2d}: KL weight = {weight:.6f}")