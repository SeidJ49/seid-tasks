def ic(*args, **kwargs):
    if not args:
        return None
    return args[0] if len(args) == 1 else args