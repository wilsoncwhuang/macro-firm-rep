from models import (
    MLP,
    FIM,
    Transformer,
    DefaultFusionNet,
    DefaultFusionNet_A3,
)

def setup_model(args):
    if args.model_type == "fim":
        print("Using FIM model")
        model = FIM.model(
            feature_size=args.feature_size,
            window_size=args.window_size,
            horizon_size=args.horizon_size,
        )
    elif args.model_type == "mlp":
        print("Using MLP model")
        model = MLP.model(
            feature_size=args.feature_size,
            window_size=args.window_size,
            horizon_size=args.horizon_size,
            dropout=args.dropout,
        )
    elif args.model_type == "transformer":
        print("Using Transformer model")
        model = Transformer.model(
            enc_in=args.feature_size,
            d_model=64,
            e_layers=3,
            n_heads=4,
            output_attention=True,
            factor=1,
            d_ff=512,
            activation="gelu",
            seq_len=args.window_size,
            num_class=args.horizon_size,
            horizon_size=args.horizon_size,
        )
    elif args.model_type == "dfn":
        print("Using Default Fusion Net model")
        model = DefaultFusionNet.model(
            feature_size=args.feature_size,
            window_size=args.window_size,
            horizon_size=args.horizon_size,
            backbone=args.backbone,
        )
    elif args.model_type == "dfna3":
        print("Using Default Fusion Net A3 model")
        model = DefaultFusionNet_A3.model(
            feature_size=args.feature_size,
            window_size=args.window_size,
            horizon_size=args.horizon_size,
        )
    else:
        print("Model type not supported")
        exit()

    return model