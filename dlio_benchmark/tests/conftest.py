# HACK: to fix the reinitialization problem
def pytest_configure(config):
    config.is_dftracer_initialized = False
