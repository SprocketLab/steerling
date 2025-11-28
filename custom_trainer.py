import torch.nn.functional as F
from transformers import Trainer
import torch

class StepwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Step-wise loss:
          - For each t, feed prefix input_ids[:, :t+1]
          - Use logits at last position to predict labels[:, t+1]
          - Only positions with labels != -100 contribute
        """
        input_ids = inputs["input_ids"]          # (B, S)
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]                # (B, S)

        B, S = input_ids.shape
        device = input_ids.device

        total_loss = 0.0
        total_count = 0
        last_outputs = None

        for t in range(S - 1):
            next_labels = labels[:, t + 1]       # (B,)
            active = next_labels != -100
            if not active.any():
                continue

            prefix_ids = input_ids[:, :t + 1]
            prefix_mask = attention_mask[:, :t + 1]

            outputs = model(
                input_ids=prefix_ids,
                attention_mask=prefix_mask,
                labels=None,
            )
            last_outputs = outputs  # keep something to return if needed

            logits = outputs.logits[:, -1, :]    # (B, V)
            logits_active = logits[active]
            labels_active = next_labels[active]

            loss_t = F.cross_entropy(
                logits_active,
                labels_active,
                reduction="sum",
            )

            total_loss = total_loss + loss_t
            total_count += labels_active.size(0)

        if total_count == 0:
            # no supervised tokens; make a 0 loss that still has grad
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = total_loss / total_count

        if return_outputs:
            # if we never entered the loop, last_outputs will be None
            return loss, last_outputs
        else:
            return loss
