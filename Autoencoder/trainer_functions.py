import torch
import torch.nn.functional as F


class Trainer():

    def __init__(self, model, optimizer, scheduler=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _train(self, train_loader):
        self.model.train()
        loss_all = 0

        for (data, target) in train_loader:

            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = F.cross_entropy(self.model(data), target)
            loss.backward()
            loss_all += loss.item() * len(data)
            self.optimizer.step()

        return loss_all / len(train_loader.dataset)

    def _test(self, loader, target_means=None, target_stds=None):
        self.model.eval()
        correct = 0

        for (data, target) in loader:

            data = data.to(self.device)

            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        return 100 * correct / len(loader.dataset)

    def run(self, train_loader, val_loader, n_epochs=300, target_means=None, target_stds=None):

        train_output = ''

        best_val_error = None
        for epoch in range(1, n_epochs + 1):

            # get learning rate from scheduler
            if self.scheduler is not None:
                lr = self.scheduler.optimizer.param_groups[0]['lr']

            loss = self._train(train_loader)

            train_error = self._test(train_loader, target_means=target_means, target_stds=target_stds)
            val_error = self._test(val_loader, target_means=target_means, target_stds=target_stds)

            # learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                best_val_error = val_error

            epoch_out_line = f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, 'f'Train Acc: {train_error:.7f}, 'f'Val Acc: {val_error:.7f}'
            print(epoch_out_line)
            train_output = epoch_out_line + '\n'

        return train_output


class AutoencoderTrainer():

    def __init__(self, model, optimizer, scheduler=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def _train(self, train_loader):
        self.model.train()
        loss_all = 0

        for (data, target) in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            loss = F.mse_loss(self.model(data), data)
            loss.backward()
            loss_all += loss.item() * len(data)
            self.optimizer.step()

        return loss_all / len(train_loader.dataset)

    def _test(self, loader, target_means=None, target_stds=None):
        self.model.eval()
        error = 0

        for (data, target) in loader:
            data = data.to(self.device)
            error += F.mse_loss(self.model(data), data).item() * len(data)

        return error / len(loader.dataset)


    def run(self, train_loader, val_loader, n_epochs=300, target_means=None, target_stds=None):

        train_output = ''

        best_val_error = None
        for epoch in range(1, n_epochs + 1):

            # get learning rate from scheduler
            if self.scheduler is not None:
                lr = self.scheduler.optimizer.param_groups[0]['lr']

            loss = self._train(train_loader)

            train_error = self._test(train_loader, target_means=target_means, target_stds=target_stds)
            val_error = self._test(val_loader, target_means=target_means, target_stds=target_stds)

            # learning rate scheduler step
            if self.scheduler is not None:
                self.scheduler.step(val_error)

            if best_val_error is None or val_error <= best_val_error:
                best_val_error = val_error

            epoch_out_line = f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, 'f'Train MSE: {train_error:.7f}, 'f'Val MSE: {val_error:.7f}'
            print(epoch_out_line)
            #output += f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, 'f'Train MAE: {train_error:.7f}, 'f'Val MAE: {val_error:.7f}' + '\n'

        return train_output
