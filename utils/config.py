config = {
    "learning_rate": 0.0003,
    "batch_size": 32,
    "epochs": 100,
    "num_classes": 5, # Change according to class - man cut: 6, man perm:5, woman cut:6, woman perm:4
    "input_size": 256,
    "data_path": "data/man_data/man_perm", # Change according to class
    "weight_decay": 1e-4,
    "model_save_path": "checkpoints/",
    "use_scheduler": True,
    "use_wandb": True

}
