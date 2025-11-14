def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0.,
        lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1,
        stop_grid_update_step=50, batch=-1,
        metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
        singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None, n_epochs=10):
    '''
    training loop with epoch, validation, metrics computation

    Args:
        n_epochs : int
            number of epochs to train
    '''
    if lamb > 0. and not self.save_act:
        print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')

    old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

    pbar = tqdm(range(n_epochs), desc='Training', ncols=100)

    if loss_fn is None:
        loss_fn = lambda x, y: torch.mean((x - y) ** 2)

    # Prepare optimizer
    if opt == "Adam":
        optimizer = torch.optim.Adam(self.get_params(), lr=lr)
    elif opt == "LBFGS":
        optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe",
                          tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

    results = {
        'train_loss': [], 'val_loss': [], 'reg': [],
    }
    if metrics is not None:
        for metric in metrics:
            results[metric.__name__] = []

    # DataLoader handling for batching
    batch_size = dataset['train_input'].shape[0] if batch == -1 or batch > dataset['train_input'].shape[0] else batch
    batch_size_test = dataset['test_input'].shape[0] if batch == -1 or batch > dataset['test_input'].shape[0] else batch

    # Metrics initialization
    best_val_loss = float('inf')
    best_state_dict = None

    # Training Loop
    for epoch in range(n_epochs):
        total_loss = 0
        self.train()  # Set the model to training mode

        for i, (x_batch, y_batch) in enumerate(
                self.create_data_loader(dataset['train_input'], dataset['train_label'], batch_size)):
            optimizer.zero_grad()
            x_batch = x_batch.to(**_TKWARGS)
            y_batch = y_batch.to(**_TKWARGS)

            y_pred = self.forward(x_batch, singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(y_pred, y_batch)

            if self.save_act:
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff) if reg_metric else 0.0
            else:
                reg_ = 0.0

            loss = train_loss + lamb * reg_
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(
            self.create_data_loader(dataset['train_input'], dataset['train_label'], batch_size))
        results['train_loss'].append(train_loss)

        # Validation step
        self.eval()  # Set the model to evaluation mode
        total_val_loss = 0

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(
                    self.create_data_loader(dataset['test_input'], dataset['test_label'], batch_size_test)):
                x_batch = x_batch.to(**_TKWARGS)
                y_batch = y_batch.to(**_TKWARGS)

                y_pred = self.forward(x_batch, singularity_avoiding=singularity_avoiding, y_th=y_th)
                val_loss = loss_fn(y_pred, y_batch)
                total_val_loss += val_loss.item()

            val_loss = total_val_loss / len(
                self.create_data_loader(dataset['test_input'], dataset['test_label'], batch_size_test))
            results['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = self.state_dict()
                torch.save(best_state_dict, f"best_model_{epoch + 1}.pt")  # Save the best model

        # Metrics computation
        if metrics is not None:
            for metric in metrics:
                metric_val = metric(self)
                results[metric.__name__].append(metric_val)

        # Display metrics
        if display_metrics is None:
            pbar.set_description(
                f"Epoch {epoch + 1}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            metric_str = ' | '.join([f'{metric}: {results[metric][-1]:.4f}' for metric in display_metrics])
            pbar.set_description(f"Epoch {epoch + 1}/{n_epochs} | {metric_str}")
    self.symbolic_enabled = old_symbolic_enabled
    return results
