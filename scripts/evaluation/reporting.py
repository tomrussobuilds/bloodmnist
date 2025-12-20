"""
Reporting Module

This module defines the structured training report and utilities for generating
final experiment summaries in Excel format, including metadata and performance metrics.
"""

# =========================================================================== #
#                                Standard Imports
# =========================================================================== #
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Sequence, Final
import logging

# =========================================================================== #
#                                Third-Party Imports
# =========================================================================== #
import pandas as pd

# =========================================================================== #
#                                Internal Imports
# =========================================================================== #
from scripts.core import Logger, Config, log_file

# Global logger instance
logger: Final[logging.Logger] = Logger().get_logger()


# =========================================================================== #
#                               EXCEL REPORTS
# =========================================================================== #

@dataclass(frozen=True)
class TrainingReport:
    """Structured data container for summarizing a complete training experiment."""
    timestamp: str
    model: str
    dataset: str
    best_val_accuracy: float
    test_accuracy: float
    test_macro_f1: float
    epochs_trained: int
    learning_rate: float
    batch_size: int
    augmentations: str
    normalization: str
    model_path: str
    log_path: str
    seed: int

    def to_vertical_df(self) -> pd.DataFrame:
        """
        Converts the report dataclass into a vertical pandas DataFrame.
        """
        data = asdict(self)
        return pd.DataFrame(list(data.items()), columns=["Parameter", "Value"]) 

    def save(self, path: Path | str) -> None:
        """Saves the report DataFrame to an Excel file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df = self.to_vertical_df()

        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            # Save vertical report
            df.to_excel(writer, sheet_name='Detailed Report', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Detailed Report']

            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BC',
                'border': 1,
                'align': 'center',
            })
            base_format = workbook.add_format({
                'border': 1,
                'align': 'left',
                'valign': 'vcenter'
            })
            wrap_format = workbook.add_format({
                'border': 1,
                'text_wrap': True,
                'valign': 'top',
                'font_size': 10
            })

            worksheet.set_column('A:A', 25, base_format)
            worksheet.set_column('B:B', 50, wrap_format)

            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)         

        logger.info(f"Training report saved (Vertical Layout) → {path}")


def create_structured_report(
    val_accuracies: Sequence[float],
    macro_f1: float,
    test_acc: float,
    train_losses: Sequence[float],
    best_path: Path,
    cfg: Config,
) -> TrainingReport:
    """
    Constructs a TrainingReport object using the final metrics and configuration.

    Args:
        val_accuracies (Sequence[float]): List of validation accuracies (to find max).
        macro_f1 (float): Final test macro F1-score.
        test_acc (float): Final test accuracy.
        train_losses (Sequence[float]): List of training losses (to count epochs).
        best_path (Path): Path to the saved model.
        cfg (Config): The configuration object used for the run.

    Returns:
        TrainingReport: The fully populated report object.
    """
    # Local import to avoid circular dependency
    from scripts.data_handler import get_augmentations_transforms

    return TrainingReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model=cfg.model_name,
        dataset=cfg.dataset_name,
        best_val_accuracy=max(val_accuracies),
        test_accuracy=test_acc,
        test_macro_f1=macro_f1,
        epochs_trained=len(train_losses),
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        augmentations=get_augmentations_transforms(cfg),
        normalization=cfg.normalization_info,
        model_path=str(best_path),
        log_path=str(log_file),
        seed=cfg.seed,
    )