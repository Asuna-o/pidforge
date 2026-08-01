from pidforge.cli import main


def test_models_command(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "fopdt" in out and "integrator" in out


def test_tune_command(capsys):
    assert main(["tune", "--plant", "fopdt:2,5,1", "--method", "simc"]) == 0
    out = capsys.readouterr().out
    assert "kp = 1.25" in out


def test_tune_all(capsys):
    assert main(["tune", "--plant", "fopdt:2,5,1", "--all"]) == 0
    out = capsys.readouterr().out
    assert "simc" in out and "cohen-coon" in out


def test_simulate_command(capsys):
    assert (
        main(
            [
                "simulate",
                "--plant",
                "fopdt:2,5,1",
                "--tuning",
                "simc",
                "--horizon",
                "40",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "Metrics:" in out and "iae" in out


def test_simulate_explicit_gains(capsys):
    assert (
        main(
            [
                "simulate",
                "--plant",
                "fopdt:1,2,0.5",
                "--tuning",
                "1:0.2",
                "--horizon",
                "20",
            ]
        )
        == 0
    )


def test_invalid_plant_fails(capsys):
    assert main(["tune", "--plant", "bogus"]) == 2
    assert "error" in capsys.readouterr().err
