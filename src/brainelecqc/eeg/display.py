"""Readable channel labels while analysis retains original derivation names."""
import re


def display_channel_name(name):
    """Show a named mastoid acquisition reference as Ref in presentation text."""
    return re.sub(r"-(?:M[12]|A[12])(?=$|[\s\].,;：，；、）)])", "-Ref", str(name), flags=re.I)


def display_channel_names(names):
    """Keep displayed channels distinct when reference aliases coincide."""
    labels = [display_channel_name(name) for name in names]
    return [f"{label} [row {i}]" if labels.count(label) > 1 else label
            for i, label in enumerate(labels)]
