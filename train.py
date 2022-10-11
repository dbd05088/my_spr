import os
import torch
from tensorboardX import SummaryWriter
from data import DataScheduler


def train_model(config, model, scheduler: DataScheduler, writer: SummaryWriter):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')
    os.makedirs(saved_model_path, exist_ok=True)
    prev_t = 0
    for step, ((x, y, corrupt, idx), t) in enumerate(scheduler):
        # Evaluate the model when task changes
        # if t != prev_t:
        if step != 0  and step % 10 == 0:
            scheduler.eval(model, writer, step + 1, eval_title='eval', task = config['data_schedule'][:t+1])

        # learn the model
        for i in range(config['batch_iter']):
            model.learn(x, y, corrupt, idx, step * config['batch_iter'] + i)

        prev_t = t

    # evlauate the model when all of takss are trained
    scheduler.eval(model, writer, step + 1, eval_title='eval', task = config['data_schedule'][:prev_t+1])
    torch.save(model.state_dict(), os.path.join(saved_model_path, 'ckpt-{}'.format(str(step + 1).zfill(6))))
    writer.flush()

