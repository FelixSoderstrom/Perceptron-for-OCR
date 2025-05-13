import os
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import transforms
import warnings

from src.models.network import MNISTClassifier
from src.data.data_loader import get_dataloaders
from src.config.hyperparameters import hyperparameters as default_hp


def define_augmentation_params(trial):
    """
    Define data augmentation hyperparameters to tune.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of augmentation parameters for this trial
    """
    # Rotation degree range
    rotation_degrees = trial.suggest_int("rotation_degrees", 0, 30)

    # Affine transformation parameters
    translate = trial.suggest_float("translate", 0.0, 0.2)
    scale = trial.suggest_float("scale", 0.8, 1.2)
    shear = trial.suggest_int("shear", 0, 20)

    # Elastic transform parameters (optional)
    use_elastic = trial.suggest_categorical("use_elastic", [True, False])
    if use_elastic:
        elastic_alpha = trial.suggest_float("elastic_alpha", 0.5, 2.0)
        elastic_sigma = trial.suggest_float("elastic_sigma", 0.3, 0.8)
    else:
        elastic_alpha = 0
        elastic_sigma = 0

    return {
        "rotation_degrees": rotation_degrees,
        "translate": translate,
        "scale": scale,
        "shear": shear,
        "use_elastic": use_elastic,
        "elastic_alpha": elastic_alpha,
        "elastic_sigma": elastic_sigma,
    }


def create_augmented_dataloaders(trial_params):
    """
    Create data loaders with trial-specific augmentation parameters.

    Args:
        trial_params: Dictionary with hyperparameters including augmentation params

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Extract augmentation parameters
    aug = trial_params.get("augmentation", {})

    # If no augmentation parameters provided, use default dataloaders
    if not aug:
        return get_dataloaders(
            batch_size=trial_params["batch_size"],
            val_batch_size=trial_params["val_batch_size"],
        )

    # Create transforms based on trial parameters
    train_transforms = []

    # Rotation if specified
    if aug.get("rotation_degrees", 0) > 0:
        train_transforms.append(
            transforms.RandomRotation(aug["rotation_degrees"])
        )

    # Affine transformations if specified
    if (
        aug.get("translate", 0) > 0
        or aug.get("scale", 1.0) != 1.0
        or aug.get("shear", 0) > 0
    ):
        train_transforms.append(
            transforms.RandomAffine(
                degrees=0,  # We already have rotation above
                translate=(aug["translate"], aug["translate"])
                if aug.get("translate", 0) > 0
                else None,
                scale=(aug["scale"], aug["scale"])
                if aug.get("scale", 1.0) != 1.0
                else None,
                shear=aug["shear"] if aug.get("shear", 0) > 0 else None,
            )
        )

    # Elastic transform if enabled
    if aug.get("use_elastic", False) and aug.get("elastic_alpha", 0) > 0:
        train_transforms.append(
            transforms.ElasticTransform(
                alpha=aug["elastic_alpha"], sigma=aug["elastic_sigma"]
            )
        )

    # Always add ToTensor and normalize at the end
    # The ToTensor transform is already applied in get_dataloaders before splitting
    train_transforms.append(transforms.Normalize((0.5,), (0.5,)))

    # Compose the transforms
    custom_train_transform = transforms.Compose(train_transforms)

    # Get the data loaders with custom transforms
    return get_dataloaders(
        batch_size=trial_params["batch_size"],
        val_batch_size=trial_params["val_batch_size"],
        custom_train_transform=custom_train_transform,
    )


def define_model_trial_params(trial):
    """
    Define hyperparameters to optimize with Optuna.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of hyperparameters for this trial
    """
    # Network architecture parameters
    conv_channels1 = trial.suggest_int("conv_channels1", 32, 128, step=16)
    conv_channels2 = trial.suggest_int("conv_channels2", 64, 256, step=32)
    conv_channels3 = trial.suggest_int("conv_channels3", 128, 512, step=64)
    kernel_size = trial.suggest_int("kernel_size", 3, 5, step=2)
    fc_size = trial.suggest_int("fc_size", 64, 512, step=64)

    # Training parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # Take the rest of the parameters from the default config
    params = default_hp.copy()

    # Update with trial-suggested values
    params.update(
        {
            "conv_channels1": conv_channels1,
            "conv_channels2": conv_channels2,
            "conv_channels3": conv_channels3,
            "kernel_size": kernel_size,
            "fc_size": fc_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout_rate": dropout_rate,
            "batch_size": batch_size,
            "val_batch_size": batch_size * 2,
        }
    )

    # Option to tune data augmentation parameters (1/3 chance to tune augmentation)
    if trial.suggest_categorical("tune_augmentation", [True, False, False]):
        augmentation_params = define_augmentation_params(trial)
        params["augmentation"] = augmentation_params

    return params


def objective(trial):
    """
    Objective function for Optuna to optimize. Trains a model with trial hyperparameters.

    Args:
        trial: Optuna trial object

    Returns:
        Validation accuracy for this trial
    """
    # Get hyperparameters for this trial
    params = define_model_trial_params(trial)

    # Initialize WandB for this trial
    wandb_logger = WandbLogger(
        project=params["wandb_project"],
        name=f"optuna-trial-{trial.number}",
        config=params,
        log_model=True,
    )

    # Define model with trial hyperparameters
    model = MNISTClassifier(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        output_size=params["output_size"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        dropout_rate=params["dropout_rate"],
        conv_channels1=params["conv_channels1"],
        conv_channels2=params["conv_channels2"],
        conv_channels3=params["conv_channels3"],
        kernel_size=params["kernel_size"],
        fc_size=params["fc_size"],
    )

    # Get data loaders with trial batch size and possibly augmentation
    train_loader, val_loader = create_augmented_dataloaders(params)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            params["checkpoints_dir"], f"trial-{trial.number}"
        ),
        filename=f"trial-{trial.number}-{{epoch}}-{{val_acc:.4f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_acc", mode="max", patience=5, verbose=True
    )

    # Configure hardware acceleration
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Setup trainer with basic callbacks
    callbacks = [checkpoint_callback, early_stopping]

    # Add our custom pruning callback
    pruning_callback = CustomPyTorchLightningPruningCallback(
        trial=trial, monitor="val_acc"
    )
    callbacks.append(pruning_callback)
    print("Using custom PyTorchLightningPruningCallback")

    # Setup trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=params["max_epochs"],
        accelerator=accelerator,
        devices=1 if accelerator != "cpu" else None,
        log_every_n_steps=500,
        deterministic=True,
    )

    # Train the model
    try:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Get the best validation accuracy
        best_val_acc = checkpoint_callback.best_model_score.item()

        # Log additional metrics to Optuna's storage
        trial.set_user_attr("best_epoch", checkpoint_callback.best_model_path)
        trial.set_user_attr(
            "best_model_path", checkpoint_callback.best_model_path
        )

        # Log the best validation accuracy to Optuna
        return best_val_acc

    except Exception as e:
        print(f"Error in trial {trial.number}: {e}")
        # Return a poor score in case of failure
        return 0.0
    finally:
        # Finish WandB run
        wandb.finish()


def run_optuna_study(
    n_trials=20, study_name="mnist-optuna-study", storage=None
):
    """
    Run an Optuna hyperparameter optimization study.

    Args:
        n_trials: Number of trials to run
        study_name: Name for this study
        storage: Optuna storage URL (None for in-memory)

    Returns:
        The Optuna study object with results
    """
    # Reduce logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Create a pruner to terminate unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=5, interval_steps=1
    )

    # Create the study
    if storage:
        print(f"Creating study with storage: {storage}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            pruner=pruner,
        )
    else:
        print("Creating in-memory study")
        study = optuna.create_study(
            study_name=study_name, direction="maximize", pruner=pruner
        )

    print(f"Starting optimization with {n_trials} trials")
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    # Print statistics
    print("Study completed!")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for param_name, param_value in study.best_params.items():
        print(f"    {param_name}: {param_value}")

    # Save the best parameters to a file
    best_params_path = os.path.join(
        default_hp["checkpoints_dir"], f"{study_name}_best_params.pt"
    )
    best_params = study.best_params
    torch.save(best_params, best_params_path)
    print(f"Best parameters saved to: {best_params_path}")

    return study


def visualize_study_results(study):
    """
    Visualize the Optuna study results.

    Args:
        study: Completed Optuna study object
    """
    try:
        from optuna.visualization import (
            plot_param_importances,
            plot_optimization_history,
            plot_intermediate_values,
            plot_contour,
            plot_edf,
            plot_slice,
        )
        import matplotlib.pyplot as plt
        import os

        # Create results directory if it doesn't exist
        results_dir = "optuna_results"
        os.makedirs(results_dir, exist_ok=True)

        # Save study information as text
        study_info_path = os.path.join(results_dir, "study_summary.txt")
        with open(study_info_path, "w") as f:
            f.write(f"Study name: {study.study_name}\n")
            f.write(f"Number of completed trials: {len(study.trials)}\n")
            f.write(f"Best trial: #{study.best_trial.number}\n")
            f.write(f"Best validation accuracy: {study.best_value:.4f}\n")
            f.write("Best hyperparameters:\n")
            for param_name, param_value in study.best_params.items():
                f.write(f"    {param_name}: {param_value}\n")

        print(f"Study summary saved to {study_info_path}")

        # Plot optimization history
        fig1 = plot_optimization_history(study)
        history_path = os.path.join(results_dir, "optuna_history.png")
        fig1.write_image(history_path)

        # Plot parameter importances
        fig2 = plot_param_importances(study)
        importances_path = os.path.join(
            results_dir, "optuna_param_importances.png"
        )
        fig2.write_image(importances_path)

        # Plot parameter relationships (contour)
        try:
            fig3 = plot_contour(study)
            contour_path = os.path.join(results_dir, "optuna_contour.png")
            fig3.write_image(contour_path)
        except Exception as e:
            print(f"Skipping contour plot: {e}")

        # Plot slices
        try:
            fig4 = plot_slice(study)
            slice_path = os.path.join(results_dir, "optuna_slice.png")
            fig4.write_image(slice_path)
        except Exception as e:
            print(f"Skipping slice plot: {e}")

        print(f"Visualization images saved to {results_dir}/")

    except ImportError:
        print(
            "Warning: Visualization requires plotly and matplotlib. Install with: pip install plotly matplotlib"
        )


# Custom Optuna callback that inherits from pl.Callback to ensure compatibility
class CustomPyTorchLightningPruningCallback(Callback):
    """
    Custom PyTorch Lightning callback for Optuna pruning that's compatible with our version.

    This is a simplified version of optuna's PyTorchLightningPruningCallback.
    """

    def __init__(self, trial, monitor):
        super().__init__()
        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer, pl_module):
        # Skip during sanity check
        if trainer.sanity_checking:
            return

        # Get current score from metrics
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                f"The metric '{self.monitor}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name."
            )
            warnings.warn(message)
            return

        # Report to Optuna
        epoch = pl_module.current_epoch
        self._trial.report(current_score.item(), step=epoch)

        # Check if we should prune
        if self._trial.should_prune():
            raise optuna.exceptions.TrialPruned(
                f"Trial was pruned at epoch {epoch}."
            )


if __name__ == "__main__":
    # Example usage:
    # Create a database storage for Optuna (SQLite in this case)
    storage_url = "sqlite:///optuna_studies.db"

    # Run the optimization
    study = run_optuna_study(
        n_trials=20, study_name="mnist-optuna-study", storage=storage_url
    )

    # Visualize the results
    visualize_study_results(study)
