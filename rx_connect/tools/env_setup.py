import resource


def set_max_open_files_limit(new_soft_limit: int = 10000) -> None:
    """Set the maximum number of open files for the current process. This is useful for
    multiprocessing, where the default limit is 1024.
    """
    # Get the current soft and hard limits for the maximum number of open files
    _, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

    # Set the new soft limit, while keeping the hard limit unchanged
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
