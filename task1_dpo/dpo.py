import torch
import torch.nn.functional as F
def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    logits = (policy_chosen_logps - policy_rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    loss = -F.logsigmoid(beta * logits).mean()
    return loss, {
        "loss": loss.detach(),
        "chosen_rewards": (beta * (policy_chosen_logps - ref_chosen_logps)).detach().mean(),
        "rejected_rewards": (beta * (policy_rejected_logps - ref_rejected_logps)).detach().mean(),
        "margins": (beta * logits).detach().mean(),
        "accuracy": (logits > 0).float().detach().mean(),
    }
