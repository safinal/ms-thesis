import torch
from tqdm import tqdm
import wandb
import data

def test_cnn(dataloader, model, return_samples=False, log=False, args=None, inferred_groups=True):
    """
    Conventional testing of a classifier.
    """
    avg_inv_acc = 0
    count = 0
    if args:
        n_classes = data.dataset_specs.datasets[args.dataset]['num_classes']
        if inferred_groups:
            n_env = 2 * data.dataset_specs.datasets[args.dataset]['num_classes']
        else:
            n_env = data.dataset_specs.datasets[args.dataset]['num_envs']
    else:
        n_classes = 2
        n_env = 4

    corrects_envs = [0]*n_env
    totals_envs = [0]*n_env
    avg_acc_envs = [0]*n_env
    worst_acc = 1

    if return_samples:
        miscls_envs = {i: [] for i in range(n_env)}
        corrcls_envs = {i: [] for i in range(n_env)}

    model.eval()
    with torch.no_grad():
        all_correct = 0
        all_totals = 0

        for (batch, (inputs, labels, envs)) in enumerate(tqdm(dataloader)):
            count += 1

            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            envs = envs.to(model.device)

            logits = model(inputs)

            all_correct += torch.sum(torch.argmax(logits, -1) == torch.argmax(labels, -1)).item()
            all_totals += inputs.shape[0]

            for env_num in range(n_env):
                logits_env = logits[envs[:, env_num] == 1]
                labels_env = labels[envs[:, env_num] == 1]
                corrects = torch.argmax(logits_env, dim=1) == torch.argmax(labels_env, dim=1)
                corrects_envs[env_num] += torch.sum(corrects).item()
                totals_envs[env_num] += len(logits_env)

                if return_samples:
                    corrcls = list(zip(inputs[envs[:, env_num] == 1][corrects], labels_env[corrects],
                                       torch.ones(labels_env[corrects].shape) * env_num))
                    corrcls_envs[env_num].extend(corrcls)
                    misclassified_indices = (corrects == 0).nonzero().flatten()
                    misclassified_samples = inputs[envs[:, env_num] == 1][misclassified_indices]
                    misclassified_labels = labels_env[misclassified_indices]
                    miscls = list(zip(misclassified_samples, misclassified_labels,
                                      torch.ones(misclassified_labels.shape) * env_num))
                    miscls_envs[env_num].extend(miscls)


    print()
    for env_num in range(n_env):
        if totals_envs[env_num] == 0:
            print(f"env {env_num}, No samples")
        else:
            avg_acc_envs[env_num] = round(corrects_envs[env_num] / totals_envs[env_num], 4)
            worst_acc = min(worst_acc, avg_acc_envs[env_num])
            print(f"env {env_num}, acc: {avg_acc_envs[env_num]}, support: {totals_envs[env_num]}")
        if log:
            wandb.log({f"Test Accuracy - Env {env_num}": avg_acc_envs[env_num]})
    avg_inv_acc = round(all_correct / all_totals, 6)

    print(f"all envs mean acc: {avg_inv_acc}")
    if log:
        wandb.log({"Test Mean Accuracy": avg_inv_acc})
    if return_samples:
        return avg_inv_acc, worst_acc, miscls_envs, corrcls_envs
    return avg_inv_acc, worst_acc