def get_dataloader_kwargs(args):
    return {
        "batch_size": args.batch_size,
        "num_workers": 10,
        "pin_memory": True,
        "shuffle": True,
        "use_macro": args.use_macro,
        "n_macro": args.n_macro,
        "level": args.level
    }


def get_openai_api_key():
    return "OPENAI_API_KEY_PLACEHOLDER"