"""
Main Execution Script for BloodMNIST Classification

This script orchestrates the entire training and evaluation pipeline:
1. Parses command-line arguments and sets up configuration.
2. Ensures environment reproducibility and prevents duplicate runs.
3. Loads the BloodMNIST dataset and creates PyTorch DataLoaders.
4. Initializes and adapts the model via the Models Factory.
5. Executes the training loop with the ModelTrainer.
6. Loads the best checkpoint and performs final evaluation, including TTA.
7. Generates visualization reports and saves structured Excel data.
"""
# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from typing import Final
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import torch

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import (
    Config, Logger, parse_args, set_seed, kill_duplicate_processes, get_device, 
    NPZ_PATH, REPORTS_DIR
)
from scripts.data_handler import (
    load_bloodmnist, get_dataloaders, show_sample_images
)
from scripts.models import get_model
from scripts.trainer import ModelTrainer
from scripts.evaluation import (
    run_final_evaluation, create_structured_report
)

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()

# =========================================================================== #
#                               MAIN EXECUTION
# =========================================================================== #

def main() -> None:
    """
    The main function that controls the entire BloodMNIST training and evaluation flow.
    """
    
    # 1. Configuration Setup
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        mixup_alpha=args.mixup_alpha,
        use_tta=not args.no_tta,
        hflip=args.hflip,
        rotation_angle=args.rotation_angle,
        jitter_val=args.jitter_val
    )
    
    # Initialize Logger and Seed
    Logger.setup(name=cfg.model_name)
    set_seed(cfg.seed)

    # 2. Environment Initialization
    kill_duplicate_processes(logger=logger)
    device = get_device(logger=logger)

    logger.info(
        f"Hyperparameters: LR={cfg.learning_rate:.4f}, Momentum={cfg.momentum:.2f}, "
        f"Batch={cfg.batch_size}, Epochs={cfg.epochs}, MixUp={cfg.mixup_alpha}, "
        f"TTA={'Enabled' if cfg.use_tta else 'Disabled'}"
    )

    # 3. Data Loading and Preparation
    data = load_bloodmnist(NPZ_PATH, cfg=cfg)
    
    # Optional: Visual check of samples (saved to figures/)
    show_sample_images(data, cfg=cfg)

    # Create DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(data, cfg)

    # 4. Model Initialization (Factory Pattern)
    # Corrected: Passing both device and cfg as required by our __init__.py factory
    model = get_model(device=device, cfg=cfg)

    # 5. Training Execution
    logger.info("Starting training pipeline".center(60, "="))

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg
    )
    best_path, train_losses, val_accuracies = trainer.train()

    # Load the best weights found during training
    model.load_state_dict(torch.load(best_path, map_location=device))
    logger.info(f"Loaded best checkpoint weights from: {best_path}")

    # 6. Final Evaluation (Metrics & Plots)
    macro_f1, test_acc = run_final_evaluation(
        model=model,
        test_loader=test_loader,
        data=data,
        train_losses=train_losses,
        val_accuracies=val_accuracies,
        device=device,
        use_tta=cfg.use_tta,
        cfg=cfg
    )

    # 7. Build and Save Structured Report
    report = create_structured_report(
        val_accuracies=val_accuracies,
        macro_f1=macro_f1,
        test_acc=test_acc,
        train_losses=train_losses,
        best_path=best_path,
        cfg=cfg
    )

    # Final Summary Logging
    logger.info(
        f"PIPELINE COMPLETED → "
        f"Test Acc: {test_acc:.4f} | "
        f"Macro F1: {macro_f1:.4f} | "
        f"Best Val Acc: {report.best_val_accuracy:.4f}"
    )

    # Save to Excel
    excel_filename = f"report_{cfg.model_name.replace(' ', '_')}.xlsx"
    report.save(REPORTS_DIR / excel_filename)
    logger.info(f"Excel report saved successfully in: {REPORTS_DIR}")


# =========================================================================== #
#                               ENTRY POINT
# =========================================================================== #

if __name__ == "__main__":
    main()