from pathlib import Path


def test_production_models_do_not_import_sklearn():
    models_source = Path("src/beijing_aqi/models.py").read_text(encoding="utf-8")

    assert "from sklearn" not in models_source
    assert "import sklearn" not in models_source
