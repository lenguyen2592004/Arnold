class DQNFeedforward(DQN):

    DQNModuleClass = DQNModuleFeedforward

    def f_train(self, screens, variables, features, actions, rewards, isfinal,
                loss_history=None):

        screens, variables, features, actions, rewards, isfinal = \
            self.prepare_f_train_args(screens, variables, features,
                                      actions, rewards, isfinal)

        batch_size = self.params.batch_size
        seq_len = self.hist_size + 1

        screens = screens.view(batch_size, seq_len * self.params.n_fm,
                               *self.screen_shape[1:])

        output_sc1, output_gf1 = self.module(
            screens[:, :-self.params.n_fm, :, :],
            [variables[:, -2, i] for i in range(self.params.n_variables)]
        )
        output_sc2, output_gf2 = self.module(
            screens[:, self.params.n_fm:, :, :],
            [variables[:, -1, i] for i in range(self.params.n_variables)]
        )

        # compute scores
        mask = torch.zeros_like(output_sc1, dtype=torch.bool)
        for i in range(batch_size):
            mask[i, int(actions[i, -1])] = 1
        scores1 = output_sc1.masked_select(mask)
        scores2 = rewards[:, -1] + (
            self.params.gamma * output_sc2.max(1)[0] * (1 - isfinal[:, -1])
        )

        # dqn loss
        loss_sc = self.loss_fn_sc(scores1, scores2.detach()).mean()

        # game features loss
        loss_gf = 0
        if self.n_features:
            loss_gf += self.loss_fn_gf(output_gf1, features[:, -2].float())
            loss_gf += self.loss_fn_gf(output_gf2, features[:, -1].float())

        self.register_loss(loss_history, loss_sc.item(), loss_gf.item())

        return loss_sc, loss_gf
