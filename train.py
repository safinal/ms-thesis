from torch import nn
from tqdm import tqdm
import torch
import mlflow

def train_cnn(dataloader, model, opt, scheduler, step, device, l1_lambda, log=True):
    criterion =  nn.CrossEntropyLoss()

    ### average loss
    avg_acc = 0
    avg_loss = 0
    count = 0

    model.train()
    for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataloader)):
        count += inputs.shape[0]

        inputs = inputs.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        logits = model(inputs)
        total_loss = criterion(logits, labels.float())
        if l1_lambda != 0:
            l1_reg = torch.tensor(0.0, requires_grad=True)
            fc = model.model.fc if hasattr(model, "model") else model.fc
            for param in fc.parameters():
                l1_reg = l1_reg + torch.norm(param, p=1)
            total_loss = total_loss + l1_lambda * l1_reg
        total_loss.backward()
        opt.step()

        avg_loss += total_loss
        avg_acc += torch.sum(torch.argmax(logits, dim=1)==torch.argmax(labels, dim=1))

    # results
    avg_acc = avg_acc/(count)
    avg_loss = avg_loss/(count)

    print("{:s}{:d}: {:s}{:.4f}, {:s}{:.4f}.".format(
        "----> [Train] Total iteration #", step, "acc: ",
        avg_acc, "loss: ", avg_loss),
          flush=True)
    if log:
        mlflow.log_metrics({"Train Accuracy": avg_acc.item() if torch.is_tensor(avg_acc) else avg_acc, "Train Loss": avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss}, step=step)

    if not scheduler==None:
        scheduler.step()

    return step+1