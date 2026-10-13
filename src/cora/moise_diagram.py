"""Compatibility entry point; implementation lives in the moise-diagram package."""
from moise_diagram import DiagramError, OrganizationDiagram, compile_file, parse, to_dot
from moise_diagram.compiler import main

__all__ = ["DiagramError", "OrganizationDiagram", "compile_file", "parse", "to_dot"]

if __name__ == "__main__":
    main()
