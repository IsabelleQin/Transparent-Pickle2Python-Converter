import argparse
import importlib

def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", 
        choices=["generate", "execute"], 
        help="Generate code from a Pickle-based checkpoint/Execute a generated Python script."
        )
    parser.add_argument(
        "framework",
        choices=["pickle", "dill", "torch", "nemo"], 
        help="Target generation/execution framework."
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        help="Path to the original Pickle-based checkpoint."
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str,
        help="Path to the code output."
    )
    parser.add_argument(
        "-mm", "--model_module",
        type=str,
        help="Model module for NeMo models."
    )
    parser.add_argument(
        "-mc", "--model_class",
        type=str,
        help="Model class for NeMo models."
    )
    # TODO: Implement this later
    parser.add_argument(
        "-conf", "--configs",
        type=str,
        help="Path to additional configurations."
    )
    args = parser.parse_args()

    if args.framework == "pickle":
        from core.pickle import generate, execute
    elif args.framework == "dill":
        from core.dill import generate, execute
    elif args.framework == "torch":
        from core.torch import generate, execute
    elif args.framework == "nemo":
        from core.nemo import generate, execute

    if args.mode == "generate":
        if not args.checkpoint:
            raise ValueError("Please provide path to the checkpoint!")
        if not args.output_path:
            raise ValueError("Please provide path for the code output!")
        try:
            generate(args.checkpoint, args.output_path)
        except Exception as e:
            raise e
        
    if args.mode == "execute":
        if not args.checkpoint and args.framework in ["torch", "nemo"]:
            raise ValueError("Please provide path to the checkpoint!")
        if not args.output_path:
            raise ValueError("Please provide path to the generated code!")
        try:
            if args.framework == "torch":
                execute(args.checkpoint, args.output_path)
            elif args.framework == "nemo":
                if not args.model_module or not args.model_class:
                    raise ValueError("Module and class import are required for NeMo model reconstruction!")
                module = importlib.import_module(args.model_module)
                model_cls = getattr(module, args.model_class)
                execute(model_cls, args.checkpoint, args.output_path)
            else:
                execute(args.output_path)
        except Exception as e:
            raise e
        
__main__()