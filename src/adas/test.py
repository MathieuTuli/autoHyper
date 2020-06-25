from typing import Tuple

import torch

from . import __globals__ as GLOBALS


def test_main(test_loader, epoch: int, device) -> Tuple[float, float]:
    # global best_acc, performance_statistics, net, criterion, checkpoint_path
    GLOBALS.NET.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = GLOBALS.NET(inputs)
            loss = GLOBALS.CRITERION(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(
            #     batch_idx, len(test_loader),
            #     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss / (batch_idx + 1), 100. * correct / total,
            #        correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > GLOBALS.BEST_ACC:
        # print('Adas: Saving checkpoint...')
        state = {
            'net': GLOBALS.NET.state_dict(),
            'acc': acc,
            'epoch': epoch + 1,
        }
        if GLOBALS.ADAS is not None:
            state['historical_io_metrics'] = GLOBALS.METRICS.historical_metrics
        torch.save(state, str(GLOBALS.CHECKPOINT_PATH / 'ckpt.pth'))
        # if checkpoint_path.is_dir():
        #     torch.save(state, str(checkpoint_path / 'ckpt.pth'))
        # else:
        #     torch.save(state, str(checkpoint_path))
        GLOBALS.BEST_ACC = acc
    GLOBALS.PERFORMANCE_STATISTICS[f'test_acc_epoch_{epoch}'] = acc / 100
    GLOBALS.PERFORMANCE_STATISTICS[f'test_loss_epoch_{epoch}'] = \
        test_loss / (batch_idx + 1)
    return test_loss / (batch_idx + 1), acc
