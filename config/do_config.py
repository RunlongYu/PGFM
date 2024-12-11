fmPgConfig = {
        'batch_size' : 60,
        'train_epochs' : 30,
        'learning_rate' : 0.001, # raw = 0.005
        'doc_threshold' : 0,
        'use_unsup' : 1,
        'lambda_total' : 1,
        'lambda_stratified_epi': 5,
        'lambda_stratified_hypo': 5,
        'use_unsup': 1,
        'lambda1': 0.0
}
fmPgPlusConfig = {
        'batch_size' : 60,
        'train_epochs' : 30,
        'learning_rate' : 0.001, # raw = 0.005
        'doc_threshold' : 0,
        'use_unsup' : 1,
        'lambda_total' : 1,
        'lambda_stratified_epi': 5,
        'lambda_stratified_hypo': 5,
        'use_unsup': 1,
        'lambda1': 0.0
}
fmLSTMConfig = {
        'batch_size' : 60,
        'train_epochs' : 15,
        'learning_rate' : 0.001, # raw = 0.005
        'doc_threshold' : 0,
        'use_unsup' : 1,
        'lambda_total' : 0,
        'lambda_stratified_epi': 0,
        'lambda_stratified_hypo': 0,
        'use_unsup': 0,
        'lambda1': 0.0
}

fmTFConfig  = {
        'batch_size' : 60,
        'train_epochs' : 25,
        'learning_rate' : 0.001, # raw = 0.005
        'doc_threshold' : 0,
        'use_unsup' : 1,
        'lambda_total' : 0,
        'lambda_stratified_epi': 0,
        'lambda_stratified_hypo': 0,
        'use_unsup': 0,
        'lambda1': 0.0
}

lstmConfig = {
        'batch_size' : 60,
        'train_epochs' : 30,
        'hidden_size' : 50,
        'learning_rate' : 0.001, # raw = 0.005
        'doc_threshold' : 0,
        'use_unsup' : 1,
        'lambda_total' : 0,
        'lambda_stratified_epi': 0,
        'lambda_stratified_hypo': 0,
        'use_unsup': 0,
        'lambda1': 0.0
}

transformerConfig = {
        'batch_size' : 60,
        'train_epochs' : 30,
        'hidden_size' : 50,
        'learning_rate' : 0.001, # raw = 0.005
        'doc_threshold' : 0,
        'use_unsup' : 1,
        'lambda_total' : 0,
        'lambda_stratified_epi': 0,
        'lambda_stratified_hypo': 0,
        'use_unsup': 0,
        'lambda1': 0.0,
        'num_heads' : 1,
        'num_layers' : 2
}

ealstmConfig = {
        'batch_size' : 60,
        'train_epochs' : 20,
        'hidden_size' : 50,
        'learning_rate' : 0.0015, # raw = 0.005
        'doc_threshold' : 0,
        'use_unsup' : 1,
        'lambda_total' : 0,
        'lambda_stratified_epi': 0,
        'lambda_stratified_hypo': 0,
        'use_unsup': 0,
        'lambda1': 0.0
}
