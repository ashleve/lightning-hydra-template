def pytest_configure(config):
    config.addinivalue_line("markers", "slow: Run slow tests.")
    config.addinivalue_line("markers", "wandb: Run the wandb tests.")
