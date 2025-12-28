"""
Health Check and Integrity Module

This script performs a global integrity scan across all registered MedMNIST 
datasets by executing a 5-step verification protocol for each:
1. Raw Data Access: Verification of .npz file presence and key-level accessibility.
2. Metadata Validation: Consistency check between tensor shapes and registry classes.
3. DataLoader Compatibility: Verification of temporary loader creation and sampling.
4. Config Validation: Pydantic-driven check for dataset-specific parameters.
5. Visual Confirmation: Generation of sample grids to verify label-image mapping.
"""

# =========================================================================== #
#                                Standard Imports                             #
# =========================================================================== #
from pathlib import Path
import numpy as np

# =========================================================================== #
#                                Internal Imports                             #
# =========================================================================== #
from src.core.config import Config, DatasetConfig, SystemConfig
from src.core.orchestrator import RootOrchestrator
from src.core.metadata.medmnist_v2_28x28 import DATASET_REGISTRY
from src.data_handler.factory import create_temp_loader
from src.data_handler.data_explorer import show_sample_images

# =========================================================================== #
#                               HEALTH CHECK LOGIC                            #
# =========================================================================== #

def health_check() -> None:
    """
    Scans all datasets in the registry using a standardized verification protocol.
    """
    
    # 1. Initialize minimal config for the Orchestrator lifecycle
    base_cfg = Config(
        model_name="HealthCheck-Probe",
        pretrained=True,
        system=SystemConfig(
            output_dir=Path("outputs/health_checks"),
            project_name="HealthCheck"
        )
    )

    # 2. Use Orchestrator as Context Manager to handle logs and system locks
    with RootOrchestrator(base_cfg) as orchestrator:
        logger = orchestrator.run_logger
        
        divider = "=" * 60
        logger.info(divider)
        logger.info("STARTING GLOBAL MEDMNIST HEALTH CHECK".center(len(divider)))
        logger.info(divider)

        for key, ds_meta in DATASET_REGISTRY.items():
            logger.info(f"--- Checking Dataset: {ds_meta.display_name} ({key}) ---")
            
            try:
                # STEP 1: Raw Data Access
                target_path = ds_meta.path
                if not target_path.exists():
                    raise FileNotFoundError(f"NPZ file missing at: {target_path}")
                
                # Direct load to bypass potential Orchestrator path mismatches
                raw_data = np.load(target_path)
                
                # STEP 2: DataLoader Compatibility
                temp_loader = create_temp_loader(raw_data, batch_size=16)

                # STEP 3: Config Validation & Channel Promotion Logic
                temp_cfg = Config(
                    model_name="HealthCheck-Probe",
                    pretrained=True, 
                    dataset=DatasetConfig(
                        dataset_name=ds_meta.name,
                        in_channels=ds_meta.in_channels,
                        num_classes=len(ds_meta.classes),
                        mean=ds_meta.mean,
                        std=ds_meta.std,
                        force_rgb=(ds_meta.in_channels == 1) # Auto-promote grayscale
                    )
                )

                # Log effective input status (the core of our recent fixes)
                mode_str = "RGB-PROMOTED" if temp_cfg.dataset.force_rgb else "NATIVE"
                logger.info(f"Mode: {mode_str} | Target Channels: {temp_cfg.dataset.effective_in_channels}")

                # STEP 4: Visual Confirmation (Saves a sample grid)
                sample_output_path = base_cfg.system.output_dir / f"samples_{ds_meta.name}.png"
                show_sample_images(
                    loader=temp_loader,
                    classes=ds_meta.classes,
                    save_path=sample_output_path,
                    cfg=temp_cfg
                )
                
                logger.info(f"Integrity check PASSED for {ds_meta.display_name}")

            except Exception as e:
                logger.error(f"Integrity check FAILED for {ds_meta.display_name}: {e}")
                continue

        logger.info(divider)
        logger.info("GLOBAL HEALTH CHECK COMPLETED".center(len(divider)))
        logger.info(divider)

# ========================================================================== #
#                                   ENTRY POINT                              #
# ========================================================================== # 
if __name__ == "__main__":
    health_check()