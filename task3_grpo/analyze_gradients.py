import torch

def main():
    print("--- Length-Conditioned Normalization Statistic ---")
    seq_adv = torch.tensor([1.0, 1.0])
    token_mask = torch.zeros(2, 768)
    token_mask[0, :50] = 1.0   # Short sequence
    token_mask[1, :500] = 1.0  # Long sequence
    
    ratio = torch.ones(2, 768)
    adv = seq_adv[:, None]
    objective = ratio * adv
    token_sum = (objective * token_mask).sum(-1)
    
    canon_denom = token_mask.sum(-1).clamp_min(1.0)
    canon_weight = token_sum / canon_denom
    print(f"Canonical Gradient Multiplier -> Short: {canon_weight[0].item():.3f} | Long: {canon_weight[1].item():.3f}")
    
    dr_denom = 768.0
    dr_weight = token_sum / dr_denom
    print(f"Dr. GRPO Gradient Multiplier -> Short: {dr_weight[0].item():.3f} | Long: {dr_weight[1].item():.3f}")

if __name__ == "__main__":
    main()
