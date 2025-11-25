from pathlib import Path
import yaml

def load_paths(cfg_file="configs/paths.yaml"):
    # Find the project root directory (where configs/ is located)
    current_dir = Path(__file__).parent.absolute()  # src/utils/
    project_root = current_dir.parent.parent  # Go up to project root
    
    # Construct the full path to the config file
    if not Path(cfg_file).is_absolute():
        cfg_file = project_root / cfg_file
    
    with open(cfg_file, "r") as f:
        p = yaml.safe_load(f)

    return {
        # Files
        "data_file": project_root / p.get("data_file", ""),
        "x_train_file": project_root / p.get("x_train_file", ""),
        "y_train_file": project_root / p.get("y_train_file", ""),

        # Directories
        "results_dir": project_root / p.get("results_dir", "results"),
        "checkpoints_dir": project_root / p.get("checkpoints_dir", "src/models"),
        "configs_dir": project_root / p.get("configs_dir", "configs"),
        "weights_dir": Path(p.get("weights_dir", "src/models")),  # Add weights_dir support
    }
