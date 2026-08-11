from torch import nn
from tqdm import tqdm
import torch


def train_cnn(dataloader, model, opt, scheduler, step, device, l1_lambda, log=True):
    criterion = nn.CrossEntropyLoss()
    
    ### average loss
    avg_acc = 0.0
    avg_loss = 0.0
    count = 0

    model.train()
    for batch, data in enumerate(tqdm(dataloader)):
        inputs, labels = data[:2]
        batch_size = inputs.shape[0]
        count += batch_size

        inputs = inputs.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        logits = model(inputs)
        
        total_loss = criterion(logits, labels.float())

        if l1_lambda != 0:
            fc = model.model.fc if hasattr(model, "model") else model.fc
            l1_reg = torch.sum(torch.abs(fc.weight))
            total_loss += l1_lambda * l1_reg
        
            
        total_loss.backward()
        opt.step()

        avg_loss += total_loss.item() * batch_size
        avg_acc += torch.sum(torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()

    # results
    avg_acc = avg_acc / count
    avg_loss = avg_loss / count

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Train] Total iteration #", step, "acc: ",
        avg_acc, "loss: ", avg_loss),
          flush=True)
          
    # if log:
    #     wandb.log({"Train Accuracy": avg_acc, "Train Loss": avg_loss})

    if scheduler is not None:
        scheduler.step()

    return step + 1