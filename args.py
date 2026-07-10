from configargparse import Parser

def parse_args():
    """Reads configuration file and returns configuration dictionary."""

    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=False,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    return args

def add_input_args(parser):
    """Loads configuration parameters into given configargparse.Parser."""

    model_names, datamodule_names = valid_model_and_datamodule_names()

    parser.add("--batch_size", type=int,
               help="The number of samples to include per batch.")

    parser.add("--data_dir", default="data",
               help="The name of the directory where data will be saved.")
    
    parser.add("--datamodule", choices=datamodule_names,
               help="The name of the DataModule to utilize.")
    
    parser.add("--eval_only", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to skip training and only evaluate the model.")
    
    parser.add("--loss", choices=["cross_entropy", "mse"], default="cross_entropy",
               help="The name of the loss function to utilize for optimization.")
    
    parser.add("--lr", type=float,
               help="The learning rate to utilize for optimization.") 
    
    parser.add("--lr_drop", default=0.1, type=float,
               help="The factor by which to drop the LR when using the step scheduler.")
    
    parser.add("--lr_scheduler", choices=["cosine", "cosine_warmup", "linear", "step"], default="step",
               help="The name of the LR scheduler to utilize.")
    
    parser.add("--lr_warmup_epochs", default=0, type=int,
               help="The number of epochs to warm up using certain schedulers.")
    
    parser.add("--model", choices=model_names,
               help="The name of the Model to utilize.")
    
    parser.add("--momentum", default=0.9, type=float,
               help="The momentum value to utilize with the SGD optimizer.")
    
    parser.add("--num_workers", default=4, type=int,
               help="The number of sub-processes to use for data loading.")
    
    parser.add("--optimizer", choices=["adam", "adamw", "sgd"], default="sgd",
               help="The name of the optimizer to utilize.")
    
    parser.add("--seed", default=1, type=int,
               help="The random seed to utilize.")
    
    parser.add("--weight_decay", default=1e-4, type=float,
               help="The l2-norm regularization to utilize during optimization.")


    return parser