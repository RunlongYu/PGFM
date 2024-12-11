General_Config = {
    'general': {
        'batch_size': 80,
        'data': -1,
        'epochs': 45,
        'validation_split': 0.1,
        'net_optim_lr': 1e-3,
        'criteo_embedding_size': 20,
        'lake_embedding_size': 5,
        'avazu_embedding_size': 40,
        'huawei_embedding_size': 15,
    },
}
CELS_Config = {
    'CELS': {
        'c': 0.9,
        'mu': 0.9,  # raw: 0.8; 0.9
        'gRDA_optim_lr': 1e-3,
        'net_optim_lr': 1e-3,
        'interaction_fc_output_dim': 5,
        # 'dnn_hidden_units': [400, 400, 32],
        'dnn_hidden_units': [128, 64, 32],
        'validation_split': 0.1,
        'mutation_threshold': 0.2,
        'mutation_probability': 0.5,
        'mutation_step_size': 5,
        'adaptation_hyperparameter': 0.99, # 0.99
        'adaptation_step_size': 10,
        'population_size': 4,
        'save_lstm_param': True,
        'lambda_1': 0.1,  # DO
        'lambda_2': 0.9,  # Temperature
        'lambda_l1': 0.0,
        'smooth_loss' : 0.0,
        'drop_out': 0.4,
        'depth_zoom': 1,
        'depth_shift': 0
    },
    'ModelFunctioning': {
        'interaction_fc_output_dim': 5,
        'dnn_hidden_units': [400, 400, 32]
    }
}
