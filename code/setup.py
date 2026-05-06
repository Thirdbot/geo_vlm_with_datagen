from pathlib import Path

root = Path(__file__).parent.parent.absolute()
storage_path = root / "storage"
storage_path.mkdir(parents=True, exist_ok=True)
source_path = root.joinpath("source")
test_img_path = root.joinpath("test")