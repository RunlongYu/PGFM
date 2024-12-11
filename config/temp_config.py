fmPgConfig = {
        'batch_size' : 40,
        'train_epochs' : 35,
        'learning_rate' : 0.001, # raw = 0.005
        'rmse_threshold' : 1.2,
        'unsup_loss_cutoff' : 40.0,
        'dc_unsup_loss_cutoff' : 1e-2,
        'lambda1' : 0.0, #magnitude hyperparameter of l1 loss
        'ec_lambda' : 0.2, #magnitude hyperparameter of ec loss
        'ec_threshold' : 36.0, #anything above this far off of energy budget closing is penalized
        'dc_lambda' : 1.0 #magnitude hyperparameter of depth-density constraint (dc) loss
}
fmLSTMConfig = {
        'batch_size' : 40,
        'train_epochs' : 20,
        'learning_rate' : 0.001, # raw = 0.005
        'rmse_threshold' : 1.2,
        'unsup_loss_cutoff' : 40.0,
        'dc_unsup_loss_cutoff' : 1e-2,
        'lambda1' : 0.0, #magnitude hyperparameter of l1 loss
        'ec_lambda' : 0.0, #magnitude hyperparameter of ec loss
        'ec_threshold' : 36.0, #anything above this far off of energy budget closing is penalized
        'dc_lambda' : 0.0 #magnitude hyperparameter of depth-density constraint (dc) loss
}
fmTFConfig = {
        'batch_size' : 40,
        'train_epochs' : 20,
        'learning_rate' : 0.001, # raw = 0.005
        'rmse_threshold' : 1.2,
        'unsup_loss_cutoff' : 40.0,
        'dc_unsup_loss_cutoff' : 1e-2,
        'lambda1' : 0.0, #magnitude hyperparameter of l1 loss
        'ec_lambda' : 0.0, #magnitude hyperparameter of ec loss
        'ec_threshold' : 36.0, #anything above this far off of energy budget closing is penalized
        'dc_lambda' : 0.0 #magnitude hyperparameter of depth-density constraint (dc) loss
}

lstmConfig = {
        'batch_size' : 40,
        'train_epochs' : 20,
        'hidden_size' : 50,
        'learning_rate' : 0.005, # raw = 0.005
        'rmse_threshold' : 1.2,
        'unsup_loss_cutoff' : 40.0,
        'dc_unsup_loss_cutoff' : 1e-2,
        'lambda1' : 0.0, #magnitude hyperparameter of l1 loss
        'ec_lambda' : 0.0, #magnitude hyperparameter of ec loss
        'ec_threshold' : 36.0, #anything above this far off of energy budget closing is penalized
        'dc_lambda' : 0.0 #magnitude hyperparameter of depth-density constraint (dc) loss
}

transformerConfig = {
        'batch_size' : 40,
        'train_epochs' : 20,
        'hidden_size' : 32,
        'learning_rate' : 0.005, # raw = 0.005
        'rmse_threshold' : 1.2,
        'unsup_loss_cutoff' : 40.0,
        'dc_unsup_loss_cutoff' : 1e-2,
        'lambda1' : 0.0, #magnitude hyperparameter of l1 loss
        'ec_lambda' : 0.0, #magnitude hyperparameter of ec loss
        'ec_threshold' : 36.0, #anything above this far off of energy budget closing is penalized
        'dc_lambda' : 0.0, #magnitude hyperparameter of depth-density constraint (dc) loss
        'num_heads' : 1,
        'num_layers' : 2
}
ealstmConfig = {
        'batch_size' : 40,
        'train_epochs' : 20,
        'hidden_size' : 32,
        'learning_rate' : 0.005, # raw = 0.005
        'rmse_threshold' : 1.2,
        'unsup_loss_cutoff' : 40.0,
        'dc_unsup_loss_cutoff' : 1e-2,
        'lambda1' : 0.0, #magnitude hyperparameter of l1 loss
        'ec_lambda' : 0.0, #magnitude hyperparameter of ec loss
        'ec_threshold' : 36.0, #anything above this far off of energy budget closing is penalized
        'dc_lambda' : 0.0, #magnitude hyperparameter of depth-density constraint (dc) loss
}