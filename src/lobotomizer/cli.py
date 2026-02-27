"""CLI entry point for lobotomizer."""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import torch
import torch.nn as nn

from lobotomizer import __version__


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="lobotomize",
        description="Composable model compression for PyTorch — make models smaller, faster, cheaper.",
    )
    parser.add_argument("--version", action="version", version=f"lobotomizer {__version__}")
    parser.add_argument("model", nargs="?", help="Path to saved model (.pt/.pth)")
    parser.add_argument("--recipe", help="Built-in recipe name or path to custom YAML")
    parser.add_argument("--prune", metavar="METHOD", help="Pruning method (l1_unstructured, random_unstructured, l1_structured, random_structured)")
    parser.add_argument("--sparsity", type=float, default=0.3, help="Pruning sparsity (default: 0.3)")
    parser.add_argument("--quantize", metavar="METHOD", help="Quantization method (dynamic, static)")
    parser.add_argument("--output", "-o", metavar="DIR", help="Output directory for compressed model")
    parser.add_argument("--input-shape", metavar="SHAPE", help='Input shape for FLOPs profiling, e.g. "1,3,224,224"')
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device (default: cpu)")
    parser.add_argument("--profile-only", action="store_true", help="Only profile the model, don't compress")
    parser.add_argument("--list-recipes", action="store_true", help="List available built-in recipes")
    return parser.parse_args(argv)


def _list_recipes() -> None:
    """Print available built-in recipes."""
    import yaml

    recipes_dir = pathlib.Path(__file__).parent / "recipes"
    if not recipes_dir.exists():
        print("No recipes directory found.")
        return

    print("Available recipes:\n")
    for f in sorted(recipes_dir.glob("*.yaml")):
        with f.open() as fh:
            data = yaml.safe_load(fh) or {}
        name = f.stem
        desc = data.get("description", "No description")
        stages = data.get("stages", [])
        stage_names = [s.get("type", "?") for s in stages]
        print(f"  {name:<20} {desc}")
        print(f"  {'':20} stages: {' → '.join(stage_names)}")
        print()


def _parse_input_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(","))


def _load_model(path: str, device: str) -> nn.Module:
    """Load a model from a .pt/.pth file."""
    warnings.warn(
        "Loading model via torch.load() with weights_only=False. "
        "Only load models you trust — pickle files can execute arbitrary code.",
        stacklevel=2,
    )
    model = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(model, nn.Module):
        print(f"Error: Expected nn.Module, got {type(model).__name__}. "
              "The file should contain a full model (not just state_dict).", file=sys.stderr)
        sys.exit(1)
    return model


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    # --list-recipes doesn't need a model
    if args.list_recipes:
        _list_recipes()
        return

    # All other commands need a model path
    if not args.model:
        print("Error: model path is required (unless using --list-recipes)", file=sys.stderr)
        sys.exit(1)

    model_path = pathlib.Path(args.model)
    if not model_path.exists():
        print(f"Error: file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    input_shape = _parse_input_shape(args.input_shape) if args.input_shape else None

    model = _load_model(str(model_path), args.device)

    # --profile-only
    if args.profile_only:
        from lobotomizer.core.profile import profile_model
        profile = profile_model(model, input_shape=input_shape, device=args.device)
        print("Model Profile:")
        print(f"  Parameters:       {profile['param_count']:,}")
        print(f"  Trainable:        {profile['param_count_trainable']:,}")
        print(f"  Size (MB):        {profile['size_mb']:.4f}")
        flops = profile.get('flops')
        print(f"  FLOPs:            {flops:,}" if flops else "  FLOPs:            N/A (provide --input-shape)")
        return

    # Build pipeline from args
    if args.recipe:
        from lobotomizer.core.recipe import build_pipeline_from_recipe
        pipeline = build_pipeline_from_recipe(args.recipe)
    elif args.prune or args.quantize:
        from lobotomizer.stages.prune import Prune
        from lobotomizer.stages.quantize import Quantize
        stages = []
        if args.prune:
            stages.append(Prune(method=args.prune, sparsity=args.sparsity))
        if args.quantize:
            stages.append(Quantize(method=args.quantize))
        from lobotomizer.core.pipeline import Pipeline
        pipeline = Pipeline(stages)
    else:
        print("Error: specify --recipe, --prune, or --quantize (or use --profile-only)", file=sys.stderr)
        sys.exit(1)

    # Run
    result = pipeline.run(model, device=args.device, input_shape=input_shape)

    # Print summary
    print(result.summary())

    # Save if requested
    if args.output:
        result.save(args.output)
        print(f"\nSaved to {args.output}/")


if __name__ == "__main__":
    main()
